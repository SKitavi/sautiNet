"""
SentiKenya Kafka Service
=========================
Handles all Kafka producer/consumer interactions.
In dev mode, provides a simulated stream for testing without Kafka.
"""

import asyncio
import json
import logging
import random
import time
from typing import AsyncGenerator, Callable, Optional, List
from datetime import datetime

from app.models.schemas import RawPost, Platform

logger = logging.getLogger(__name__)


# ── Simulated data for development ──

SIMULATED_TEXTS = {
    "en": [
        "The government needs to address the rising cost of living immediately",
        "Great progress on the road construction in Kiambu County",
        "Healthcare workers deserve better pay and working conditions",
        "The new education curriculum is confusing parents and teachers alike",
        "M-Pesa has revolutionized how we do business in Kenya",
        "Corruption continues to drain public resources meant for development",
        "The housing levy is adding unnecessary burden on workers",
        "Kenya's tech ecosystem is growing faster than ever before",
        "Fuel prices keep going up while salaries remain stagnant",
        "The judiciary must remain independent from political interference",
        "Our farmers need better access to markets and fair prices",
        "Security has improved significantly in the northern counties",
        "Youth unemployment is a ticking time bomb we must address",
        "The new expressway has reduced travel time significantly",
        "Water scarcity in arid counties needs urgent government attention",
    ],
    "sw": [
        "Serikali inafanya kazi nzuri katika ujenzi wa barabara",
        "Hali ya uchumi ni mbaya sana kwa wananchi wa kawaida",
        "Tunahitaji mabadiliko katika sekta ya elimu nchini Kenya",
        "Vijana wanahitaji kazi na fursa za kujiajiri",
        "Elimu ni muhimu sana kwa maendeleo ya taifa letu",
        "Maji ni uhai lakini bado kuna maeneo bila maji safi",
        "Barabara zetu ni mbaya na zinahitaji kutengenezwa",
        "Afya bora kwa wote ni haki ya kila Mkenya",
        "Ushuru umezidi na unaumiza wafanyabiashara wadogo",
        "Serikali imeshindwa kupunguza bei ya mafuta",
        "Rushwa ni adui mkubwa wa maendeleo katika nchi yetu",
        "Walimu wetu wanastahili mshahara bora zaidi",
        "Kilimo kinaweza kuwa suluhisho la ukosefu wa ajira",
        "Teknolojia inabadilisha jinsi tunavyofanya biashara Kenya",
    ],
    "sh": [
        "Siste hii economy ni tight sana hakuna pesa",
        "Manze serikali iko rada lakini watu hawajui",
        "Hii mambo ya tax ni noma bana inaumiza wasee",
        "Youth tunahitaji chance ya kufanya hustle",
        "Nairobi inakuanga busy sana maze haiwezekani",
        "Pesa imeisha bana hii life ni struggle tu",
        "Hii life ni noma lakini tuko rada wasee",
        "Tech ndio future ya Kenya na youth wako ready",
        "Manze bei ya mafuta imepanda tena si poa",
        "Wasee wa bunge wanajipangia salary kubwa na sisi tunapiga hustle",
        "Hii serikali inafanya form tu hawaskii wananchi",
        "Cheki vile Safaricom wamebadilisha game ya tech Kenya",
    ],
}

SIMULATED_COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu",
    "Machakos", "Kakamega", "Kilifi", "Meru", "Nyeri",
    "Uasin Gishu", "Bungoma", "Garissa", "Turkana", "Nandi",
]

SIMULATED_HASHTAGS = [
    ["KenyaDecides", "Elections2027"], ["TaxReform", "KenyaFinance"],
    ["Healthcare4All", "UHC"], ["DigitalKenya", "TechKE"],
    ["YouthEmpowerment", "Ajira"], ["RoadConstruction", "Infrastructure"],
    ["WaterIsLife", "MajiNiUhai"], ["EducationReform", "CBC"],
    ["FuelPrices", "CostOfLiving"], ["AntiCorruption", "EACC"],
]


class KafkaService:
    """
    Kafka producer/consumer service.

    In production: Connects to Apache Kafka cluster.
    In development: Simulates a stream of Kenyan social media posts.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        simulate: bool = True,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.simulate = simulate
        self._producer = None
        self._consumer = None
        self._running = False

    async def connect(self):
        """Connect to Kafka cluster (or initialize simulation)."""
        if self.simulate:
            logger.info("Kafka service running in SIMULATION mode")
            self._running = True
            return

        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
            )
            await self._producer.start()
            logger.info(f"Kafka producer connected to {self.bootstrap_servers}")
            self._running = True

        except ImportError:
            logger.warning("aiokafka not installed, falling back to simulation mode")
            self.simulate = True
            self._running = True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.simulate = True
            self._running = True

    async def produce(self, topic: str, message: dict):
        """Send a message to a Kafka topic."""
        if self.simulate:
            logger.debug(f"[SIM] Produced to {topic}: {message.get('post_id', 'unknown')}")
            return

        if self._producer:
            await self._producer.send_and_wait(topic, message)

    async def consume_stream(
        self,
        topic: str = "sentikenya.raw.posts",
        interval: float = 1.5,
    ) -> AsyncGenerator[RawPost, None]:
        """
        Consume posts from Kafka topic as an async generator.

        In simulation mode, generates realistic Kenyan social media posts.
        """
        if self.simulate:
            async for post in self._simulated_stream(interval):
                yield post
            return

        try:
            from aiokafka import AIOKafkaConsumer

            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id="sentikenya-nlp-workers",
                value_deserializer=lambda v: json.loads(v.decode()),
            )
            await consumer.start()
            logger.info(f"Kafka consumer started on topic: {topic}")

            try:
                async for message in consumer:
                    try:
                        post = RawPost(**message.value)
                        yield post
                    except Exception as e:
                        logger.error(f"Failed to parse message: {e}")
            finally:
                await consumer.stop()

        except ImportError:
            logger.warning("aiokafka not available, using simulation")
            async for post in self._simulated_stream(interval):
                yield post

    async def _simulated_stream(
        self,
        interval: float = 1.5,
    ) -> AsyncGenerator[RawPost, None]:
        """Generate simulated Kenyan social media posts."""
        post_counter = 0

        while self._running:
            # Random language selection (weighted toward English)
            lang = random.choices(["en", "sw", "sh"], weights=[0.45, 0.35, 0.20])[0]
            text = random.choice(SIMULATED_TEXTS[lang])

            # Add some variation
            if random.random() < 0.3:
                text += " " + " ".join(
                    f"#{tag}" for tag in random.choice(SIMULATED_HASHTAGS)
                )

            platform = random.choice(list(Platform))
            county = random.choice(SIMULATED_COUNTIES)

            post = RawPost(
                post_id=f"sim_{post_counter}_{int(time.time())}",
                platform=platform,
                text=text,
                author_handle=f"@user_{random.randint(1000, 9999)}",
                timestamp=datetime.utcnow(),
                county=county,
                engagement={
                    "likes": random.randint(0, 5000),
                    "retweets": random.randint(0, 2000),
                    "replies": random.randint(0, 500),
                    "shares": random.randint(0, 1000),
                },
                hashtags=random.choice(SIMULATED_HASHTAGS) if random.random() < 0.4 else [],
            )

            yield post
            post_counter += 1

            # Variable interval for realism
            jitter = random.uniform(0.5, 2.0) * interval
            await asyncio.sleep(jitter)

    async def disconnect(self):
        """Disconnect from Kafka."""
        self._running = False
        if self._producer:
            await self._producer.stop()
        logger.info("Kafka service disconnected")
