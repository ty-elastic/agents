from elasticsearch import Elasticsearch, helpers
import os
from typing import Optional, Union
from typing import AsyncIterable, List, Optional
from datetime import datetime, timezone, timedelta
import traceback
import asyncio
from enum import Enum

from chatgpt import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
)

TranscriptionType = Enum("TranscriptionType", ["Chat", "Spoken"])

SYSTEM_PROMPT = "You are Recall, a friendly voice assistant powered by Elastic. Please use only the results returned from your function call to answer questions."

def make_prompt(query, context):
    context = ", ".join(context)
    prompt = f"Please answer the following question '{query}' \
            using only the following context: '{context}'. If you are asked about something somone said, please assume the context \
            you are given are clauses spoken by that person. Please keep your response brief."
    return prompt

class ElasticTranscriptPlugin:
    def __init__(self, 
                 *,
                 es_username: str, 
                 es_password: str,
                 es_endpoint: str):
        self._es_username = es_username
        self._es_password = es_password
        self._es_endpoint = es_endpoint

        self._assistant= ChatGPTPlugin(
            prompt=SYSTEM_PROMPT, message_capacity=20, model="gpt-4-32k", tools=self.get_tools(), call_function=self.call_function
        )

        print(self._es_endpoint)

        self._setup()

        self._index_queue = asyncio.Queue()
        self._index_task = asyncio.create_task(self._run_index())

        self._query_queue = asyncio.Queue()
        self._query_task = asyncio.create_task(self._run_query())

    async def make_assistant_stream(self, query) -> AsyncIterable[str]:

        # start = query.lower().find("about") + len("about")
        # if query[start] == ' ': start = start+1
        # subject = query[start:]
        # if subject[len(subject)-1] == '?':
        #     subject = subject[:len(subject)-1]
        # print(subject)
        
        # context = self.query_transcripts(query=subject)
        # print(context)

        # prompt = make_prompt(query, context)
        # print(prompt)

        msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=query)
        stream = self._assistant.add_message(msg)
        async for thing in stream:
            if isinstance(thing, str):
                yield thing
            elif isinstance(thing, object):
                call = thing
                print(call)
                if call['name'] == 'get_summary_of_topic':
                    results = self.query_transcripts(query=call['arguments']['topic'])
                    #messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
                    msg2 = ChatGPTMessage(role=ChatGPTMessageRole.function, content=", ".join(results), id=call['id'], name=call['name'])
                    print(msg2)
                    stream2 = self._assistant.add_message(msg2)
                    async for thing2 in stream2:
                        if isinstance(thing2, str):
                            yield thing2
                        elif isinstance(thing2, object):
                            call2 = thing2
                            if call2['name'] == 'store_in_knowledgebase':
                                results = self.store_in_knowledgebase(knowledge=call2['arguments']['knowledge'])
                                yield "ok, I stored that information in the knowledgebase."
                elif call['name'] == 'store_in_knowledgebase':
                        results = self.store_in_knowledgebase(knowledge=call['arguments']['knowledge'])
                        yield "ok, I stored that information in the knowledgebase."
        return 

    def _setup(self):
            with Elasticsearch(self._es_endpoint, basic_auth=(self._es_username, self._es_password)) as es:
                try:
                    es.ingest.put_pipeline(
                        id="talktome-clauses",
                        description="Ingest pipeline for talktome-clauses",
                        processors=[
                            {
                                "inference": {
                                    "model_id": ".elser_model_2_linux-x86_64",
                                    "input_output": [
                                        {"input_field": "clause.text", "output_field": "clause.sparse"}
                                    ],
                                }
                            }
                        ],
                    )
                except Exception as inst:
                    print(f"unable to create indices:{inst}")      

                try:
                    es.ingest.put_pipeline(
                        id="talktome-knowledgebase",
                        description="Ingest pipeline for talktome-knowledgebase",
                        processors=[
                            {
                                "inference": {
                                    "model_id": ".elser_model_2_linux-x86_64",
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
                        index="talktome-clauses",
                        settings={"index": {"default_pipeline": "talktome-clauses"}},
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
                        index="talktome-knowledgebase",
                        settings={"index": {"default_pipeline": "talktome-knowledgebase"}},
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

    def get_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_summary_of_topic",
                    "description": "Get a summary of what someone said about a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speaker": {
                                "type": "string",
                                "description": "The name of the person who said something",
                            },
                            "topic": {
                                "type": "string",
                                "description": "the topic they talked about",
                            }
                        },
                        "required": ["topic"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "store_in_knowledgebase",
                    "description": "Store knowledge in a knowledgebase",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "knowledge": {
                                "type": "string",
                                "description": "the knowledge to store in the knowlegebase",
                            }
                        },
                        "required": ["knowledge"],
                    },
                },
            }
        ]
        return tools

    async def _run_index(self) -> None:
        while True:
            transcript_record = await self._index_queue.get()
            print("GOT RECORD")
            self._index_transcript(transcript_record)

    async def _run_query(self) -> None:
        while True:
            call = await self._query_queue.get()
            if call['name'] == 'get_summary_of_topic':
                results = self.query_transcripts(query=call['arguments']['topic'])
                #messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
                msg = ChatGPTMessage(role=ChatGPTMessageRole.function, content=results, id=call['id'], name=call['name'])
                print(msg)
                return self._assistant.add_message(msg)
            elif call['name'] == 'store_in_knowledgebase':
                results = self.store_in_knowledgebase(query=call['arguments']['knowledge'])
                #messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
                #msg = ChatGPTMessage(role=ChatGPTMessageRole.function, content=results, id=call['id'], name=call['name'])
                print(msg)
                #return self._assistant.add_message(msg)
            
            

    def push_transcript(self, *, type: TranscriptionType, clause: str, timestamp: datetime, participant_name: str, conference_id: str):
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
        print("PUT RECORD")
    
    def call_function(self, function):
        self._query_queue.put_nowait(function)
        print("PUT FUNC")

    def store_in_knowledgebase(self, *,knowledge):
        print("store_in_knowledgebase")
        knowledge_record={
            '@timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'knowledge': {
                'text': knowledge
            }
        }

        try:
            with Elasticsearch(self._es_endpoint, basic_auth=(self._es_username, self._es_password)) as es:
                resp = es.index(
                    index="talktome-knowledgebase",
                    document=knowledge_record)
                print(resp['_id'])
        except Exception as inst:
            print(f"unable to create knowledge:{inst}")
            traceback.print_exc()
            return None  


    def query_transcripts(self, *, participant_name: str = None, query: str):
        try:
            with Elasticsearch(self._es_endpoint, basic_auth=(self._es_username, self._es_password)) as es:
                response = es.search(
                    index="talktome-clauses",
                    source=False,
                    fields=['clause.text', 'clause.sparse'],
                    size=5,
                    query={
                        "text_expansion": {
                            "clause.sparse": {
                                "model_id": ".elser_model_2_linux-x86_64",
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

    def _index_transcript(self, transcript_record: object):
        try:
            with Elasticsearch(self._es_endpoint, basic_auth=(self._es_username, self._es_password)) as es:
                resp = es.index(
                    index="talktome-clauses",
                    document=transcript_record)
                print(resp['_id'])
        except Exception as inst:
            print(f"unable to create transcript:{inst}")
            traceback.print_exc()
            return None

async def test():
    plugin = ElasticTranscriptPlugin(
            es_username=os.environ.get("ELASTIC_TRANSCRIPT_USERNAME"),
            es_password=os.environ.get("ELASTIC_TRANSCRIPT_PASSWORD"),
            es_endpoint=os.environ.get("ELASTIC_TRANSCRIPT_ENDPOINT")
        )


    assistant_stream = plugin.make_assistant_stream("summarize what Ty said about apples and store it in the knowledgebase")
    async for text in assistant_stream:
        print("ty" + text)
    print('done')
    await asyncio.sleep(5)
    


#asyncio.run(test())
