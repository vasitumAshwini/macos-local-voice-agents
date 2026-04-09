import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from tts_mlx_isolated import TTSMLXIsolated

load_dotenv(override=True)

app = FastAPI()
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

SYSTEM_INSTRUCTION = """
You are Pipecat, a friendly, helpful chatbot.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so do not include markdown or special formatting.

Respond briefly unless the user asks for detail. Prefer one or two short sentences.

Latency behavior:
- If you need a moment, start with a short filler such as "Hmm..." or "Okay..." and then continue.
- Keep filler words very short and only use them when needed.
- Do not overuse filler words.

Start the conversation by saying, "Hello, I'm Pipecat!" Then stop and wait for the user.
""".strip()


async def run_bot(webrtc_connection):
    # VAD is the first gate. 0.35s is a practical "tight but not too tight" silence threshold.
    # Pipecat's default smart-turn guidance still recommends short silence windows, and smart turn
    # works best when VAD stop_secs is kept low. :contentReference[oaicite:1]{index=1}
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.35)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(
                smart_turn_model_path="",  # downloaded from HuggingFace cache
                params=SmartTurnParams(),
            ),
        ),
    )

    # NOTE:
    # Whisper MLX here is still a turn-final ASR path, not true partial-transcript streaming.
    # For real multi-stage streaming STT, replace this with a streaming STT backend.
    stt = WhisperSTTServiceMLX(
        model=MLXModel.BASE, # Faster than LARGE on M1 for voice
        device="gpu"         # Explicitly target the GPU
    )

    # NOTE:
    # This TTS is local and simple, but not a true chunked/streaming TTS engine.
    # For real dual-streaming TTS, replace with a streaming TTS service.
    tts = TTSMLXIsolated(
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_heart",
        sample_rate=24000,
    )

    llm = OpenAILLMService(
        api_key="dummyKey",
        model="gemma-4-e4b-it", # Latest and greatest model. Uses ~8GB of RAM. as of APR 2026.
        #model="gemma-3n-e4b-it-text",  # Small model. Uses ~4GB of RAM.
        # model="google/gemma-3-12b",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url="http://127.0.0.1:1234/v1",
        max_tokens=4096,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            }
        ],
    )

    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            # Keep sample formats aligned to reduce hidden resampling overhead.
            # If your transport / TTS backend expects other rates, update them together.
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", default="localhost", help="Host for HTTP server")
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)