import traceback
import os
import asyncio
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import dateutil.parser

from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch, Elasticsearch

TranscriptionSource = Enum("TranscriptionSource", ["Spoken"])
SemanticModel = Enum("SemanticModel", ["ELSER", "External"])

ELSER_MODEL_ID = ".elser_model_2_linux-x86_64"
EXTERNAL_SEMANTIC_MODEL_ID = "all-mpnet-base-v2"
EXTERNAL_SEMANTIC_MODEL_DIMS = 768

TRANSCRIPTION_INDEX_POSTFIX = "clauses"
MAX_SEARCH_RESULTS = 4

@dataclass
class ElasticRagPluginOptions:
    es_apikey: str
    es_endpoint: str
    es_index_prefix: str

class ElasticRagPlugin:
    def __init__(self, 
                 *,
                 es_apikey: str, 
                 es_endpoint: str,
                 es_index_prefix: str = "conferences",
                 semantic_model: SemanticModel = SemanticModel.ELSER,
                 start: bool = True,
                 setup: bool = False):

        self._config = ElasticRagPluginOptions(
            es_apikey=es_apikey,
            es_endpoint=es_endpoint,
            es_index_prefix=es_index_prefix
        )

        self._closed = False
        self._semantic_model = semantic_model
        
        if self._semantic_model == SemanticModel.External:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self._model = SentenceTransformer(EXTERNAL_SEMANTIC_MODEL_ID)
            
        if setup:
            self._setup()
        if start:
            self._index_queue = asyncio.Queue()
            self._index_task = asyncio.create_task(self._run_index())

    def _setup(self):
        try:
            es = Elasticsearch(self._config.es_endpoint, api_key=self._config.es_apikey)
            es.indices.delete(index=f"{self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}")
        except Exception:
            pass
        
        if self._semantic_model == SemanticModel.ELSER:
            es = Elasticsearch(self._config.es_endpoint, api_key=self._config.es_apikey)
            try:
                es.ingest.put_pipeline(
                    id=f"{self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}",
                    description=f"Ingest pipeline for {self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}",
                    processors=[
                        {
                            "inference": {
                                "model_id": ELSER_MODEL_ID,
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
            mappings={
                    "properties": {
                        "clause.text": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                        },
                        "speaker.name.text": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                            "analyzer": "standard"
                        },
                        "conference.id": {
                            "type": "keyword"
                        },
                        "source": {
                            "type": "keyword"
                        }
                    }
                }
            
            settings=None
            if self._semantic_model == SemanticModel.ELSER:
                settings={"index": {"default_pipeline": f"{self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}"}}
                mappings["properties"]["clause.sparse"] = {"type": "sparse_vector"}
            else:
                mappings["properties"]["clause.dense"] = {"type": "dense_vector", "dims": EXTERNAL_SEMANTIC_MODEL_DIMS}
            
            es.indices.create(
                index=f"{self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}",
                settings=settings,
                mappings=mappings
            )
        except Exception as inst:
            #traceback.print_exc()
            print(f"unable to create indices:{inst}")

    def push_transcript(self, *, transcription_source: TranscriptionSource, clause: str, 
                        timestamp: datetime, speaker_name: str, conference_id: str):
        if self._closed:
            return
        
        transcript_record={
            '@timestamp': timestamp.isoformat(),
            'speaker.name': {
                'text': speaker_name
            },
            'source': transcription_source.name,
            'clause': {
                'text': clause
            },
            'conference.id': conference_id
        }
        self._index_queue.put_nowait(transcript_record)
    
    async def _run_index(self) -> None:
        while not self._closed:
            transcript_record = await self._index_queue.get()
            if transcript_record is None:
                break
            await self._index_transcript(transcript_record)
            self._index_queue.task_done()
            
    def close(self):
        self._closed = True
        self._index_queue.put_nowait(None)

    def conform_search_result(self, result):
        clause = {}
        for key in result:
            if '.sparse' in key:
                continue
            value = result[key]
            if isinstance(result[key], list) and len(result[key]) > 0:
                value = result[key][0]
            if key in ['@timestamp']:
                clause[key] = dateutil.parser.isoparse(value)
            else:
                clause[key] = value
        return clause
    
    async def query_transcripts(self, *, query: str, speaker_name: str = None, conference_id: str = None):
        
        def setup_query():
            block = {}
            block.setdefault('query', {}).setdefault('bool', {}).setdefault('must', [])
            if conference_id is not None:
                block['query']['bool']['must'].append({ "match": { "conference.id": conference_id }})
            if speaker_name is not None and speaker_name != "*":
                block['query']['bool']['must'].append({ "match": { "speaker.name.text": speaker_name }}) 
            return block
        
        try:
            es_query = {}
            es_query.setdefault('sub_searches', [])
            
            bm25 = setup_query()
            bm25['query']['bool']['must'].append({ "match": { "clause.text": query }})
            es_query['sub_searches'].append(bm25)
            
            if self._semantic_model == SemanticModel.ELSER:
                elser = setup_query()
                elser['query']['bool']['must'].append(
                    {"text_expansion": {    
                        "clause.sparse": {
                            "model_id": ELSER_MODEL_ID,
                            "model_text": query
                        }
                    }})
                es_query['sub_searches'].append(elser)
            else:
                external = setup_query()
                task = asyncio.create_task(asyncio.to_thread(self._model.encode, query))
                await task
                external['query']['bool']['must'].append(
                    {"knn": {
                        "field": "clause.dense",
                        "query_vector": task.result().tolist(),
                        "num_candidates": 10
                    }})
                es_query['sub_searches'].append(external)

            es = AsyncElasticsearch(self._config.es_endpoint, api_key=self._config.es_apikey)
            response = await es.search(
                index=f"{self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}",
                source=False,
                fields=['clause.text', 'speaker.name.text', '@timestamp', 'conference.id'],
                size=MAX_SEARCH_RESULTS,
                body=es_query,
                rank={"rrf": {}},
            )
            await es.close()
            clauses = []
            for hit in response["hits"]["hits"]:
                clause = self.conform_search_result(hit['fields'])
                clauses.append({'quote': clause['clause.text'], 'speaker': clause['speaker.name.text']})

            return clauses
        except Exception as inst:
            print(f"unable to query transcript:{inst}")
            traceback.print_exc()
            return []

    async def _index_transcript(self, transcript_record: object):
        try:
            es = AsyncElasticsearch(self._config.es_endpoint, api_key=self._config.es_apikey)
            
            if self._semantic_model == SemanticModel.External:
                task = asyncio.create_task(asyncio.to_thread(self._model.encode, transcript_record['clause']['text']))
                await task
                transcript_record['clause.dense'] = task.result().tolist()
            
            await es.index(
                index=f"{self._config.es_index_prefix}-{TRANSCRIPTION_INDEX_POSTFIX}",
                document=transcript_record)
            await es.close()
        except Exception as inst:
            print(f"unable to create transcript:{inst}")
            traceback.print_exc()
            return None
