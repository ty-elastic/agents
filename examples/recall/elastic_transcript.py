import traceback
import asyncio
from enum import Enum
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch, Elasticsearch

TranscriptionSource = Enum("TranscriptionSource", ["Chat", "Spoken"])

SEMANTIC_MODEL_ID = ".elser_model_2_linux-x86_64"

@dataclass
class ElasticTranscriptPluginOptions:
    es_username: str
    es_password: str
    es_endpoint: str
    es_index_prefix: str

class ElasticTranscriptPlugin:
    def __init__(self, 
                 *,
                 es_username: str, 
                 es_password: str,
                 es_endpoint: str,
                 es_index_prefix: str):

        self._config = ElasticTranscriptPluginOptions(
            es_username=es_username,
            es_password=es_password,
            es_endpoint=es_endpoint,
            es_index_prefix=es_index_prefix
        )

        self._setup()

        self._index_queue = asyncio.Queue()
        self._index_task = asyncio.create_task(self._run_index())

    def _setup(self):
        es = Elasticsearch(self._config.es_endpoint, basic_auth=(self._config.es_username, self._config.es_password))
        try:
            es.ingest.put_pipeline(
                id=f"{self._config.es_index_prefix}-clauses",
                description=f"Ingest pipeline for {self._config.es_index_prefix}-clauses",
                processors=[
                    {
                        "inference": {
                            "model_id": SEMANTIC_MODEL_ID,
                            "input_output": [
                                {"input_field": "clause.text", "output_field": "clause.sparse"}
                            ],
                        }
                    }
                ]
            )
        except Exception as inst:
            print(f"unable to create indices:{inst}")      

        try:
            es.ingest.put_pipeline(
                id=f"{self._config.es_index_prefix}-knowledgebase",
                description=f"Ingest pipeline for {self._config.es_index_prefix}-knowledgebase",
                processors=[
                    {
                        "inference": {
                            "model_id": SEMANTIC_MODEL_ID,
                            "input_output": [
                                {"input_field": "knowledge.text", "output_field": "knowledge.sparse"}
                            ],
                        }
                    }
                ],
            )
        except Exception as inst:
            print(f"unable to create indices:{inst}")               

        try:
            es.indices.create(
                index=f"{self._config.es_index_prefix}-clauses",
                settings={"index": {"default_pipeline": f"{self._config.es_index_prefix}-clauses"}},
                mappings={
                    "properties": {
                        "clause.text": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "clause.sparse": {"type": "sparse_vector"},
                    }
                },
            )
        except Exception as inst:
            print(f"unable to create indices:{inst}")   

        try: 
            es.indices.create(
                index=f"{self._config.es_index_prefix}-knowledgebase",
                settings={"index": {"default_pipeline": f"{self._config.es_index_prefix}-knowledgebase"}},
                mappings={
                    "properties": {
                        "knowledge.text": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "knowledge.sparse": {"type": "sparse_vector"},
                    }
                },
            )
        except Exception as inst:
            print(f"unable to create indices:{inst}")   

    def push_transcript(self, *, transcriptionSource: TranscriptionSource, clause: str, timestamp: datetime, participant_name: str, conference_id: str):
        transcript_record={
            '@timestamp': timestamp.isoformat(),
            'participant_name': participant_name,
            'type': type.name,
            'clause': {
                'text': clause
            },
            'conference_id': conference_id
        }
        self._index_queue.put_nowait(transcript_record)
    
    async def query_transcripts(self, *, participant_name: str = None, query: str):
        try:
            es = AsyncElasticsearch(self._config.es_endpoint, basic_auth=(self._config.es_username, self._config.es_password))
            response = await es.search(
                index=f"{self._config.es_index_prefix}-clauses",
                source=False,
                fields=['clause.text', 'clause.sparse'],
                size=5,
                query={
                    "text_expansion": {
                        "clause.sparse": {
                            "model_id": SEMANTIC_MODEL_ID,
                            "model_text": query,
                        }
                    }
                },
            )
            clauses = []
            print(response)
            for hit in response["hits"]["hits"]:
                print(hit)
                clauses.append(hit["fields"]["clause.text"][0])
            return clauses
        except Exception as inst:
            print(f"unable to create transcript:{inst}")
            traceback.print_exc()
            return None

    async def _run_index(self) -> None:
        while True:
            transcript_record = await self._index_queue.get()
            self._index_transcript(transcript_record)

    def _index_transcript(self, transcript_record: object):
        try:
            with Elasticsearch(self._config.es_endpoint, basic_auth=(self._config.es_username, self._config.es_password)) as es:
                resp = es.index(
                    index=f"{self._config.es_index_prefix}-clauses",
                    document=transcript_record)
                print(resp['_id'])
        except Exception as inst:
            print(f"unable to create transcript:{inst}")
            traceback.print_exc()
            return None
