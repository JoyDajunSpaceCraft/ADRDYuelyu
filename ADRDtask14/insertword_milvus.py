# hello_milvus.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
import time
import pandas as pd
import numpy as np
import json
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/embeddings.csv")
posts = df.post.values.tolist()
titles = df.title.values.tolist()
embeddings = df.embeding.values.tolist()
for idx, i in enumerate(embeddings):
    i = i.split("[")[1].split("]")[0]
    arrs = np.fromstring(i, dtype=float, sep='\t')
    embeddings[idx] = arrs

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 10076, 384

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")

#################################################################################
# 2. create collection
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")

################################################################################
# 3. insert data
entities = [
    # provide the pk field because `auto_id` is set to False
    [i for i in range(num_entities)],
    embeddings,  # field embeddings, supports numpy.ndarray and list
]

insert_result = hello_milvus.insert(entities)

print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for hello_milvus collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
hello_milvus.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
med_drive = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/med_drive.csv")
drive = med_drive.drive.values.tolist()
med = med_drive.med.values.tolist()
drive = np.asarray(drive)
med = np.asarray(med)
# single ordered
memory_loss = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/memory_loss_embedding.csv")
s1 = memory_loss.s1.values.tolist()
s2 = memory_loss.s2.values.tolist()
s3 = memory_loss.s3.values.tolist()
s4 = memory_loss.s4.values.tolist()
s1 = np.asarray(s1)
s2 = np.asarray(s2)
s3 = np.asarray(s3)
s4 = np.asarray(s4)

# shuffle ordered
# memory_loss_shuffle = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/memory_loss_shuffle_embedding.csv")
# order1234 = memory_loss_shuffle["order1234"].values.tolist()
# order2134 = memory_loss_shuffle["order2134"].values.tolist()
# order3124 = memory_loss_shuffle["order3124"].values.tolist()
# order4321 = memory_loss_shuffle["order4321"].values.tolist()

# medical and drive
vectors_to_search_drive = [drive]
vectors_to_search_med = [med]

# single 1-4
vectors_to_search_s1 = [s1]
vectors_to_search_s2 = [s2]
vectors_to_search_s3 = [s3]
vectors_to_search_s4 = [s4]

# shuffle 1-4
# vectors_to_search_order1234 = [order1234]
# vectors_to_search_order2134 = [order2134]
# vectors_to_search_order3124 = [order3124]
# vectors_to_search_order4321 = [order4321]

search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 20},
}

start_time = time.time()

# all_search_items = [vectors_to_search_s1, vectors_to_search_s2, vectors_to_search_s3, vectors_to_search_s4,
#                     vectors_to_search_order1234, vectors_to_search_order2134, vectors_to_search_order3124,
#                     vectors_to_search_order4321]
all_search_items = [vectors_to_search_s1, vectors_to_search_s2, vectors_to_search_s3, vectors_to_search_s4]
store_json = {}
# name_list = ["s1","s2","s3","s4","order1234","order2134","order3124","order4321"]
name_list = ["s1","s2","s3","s4"]

## this time is not only in the order but also the score of each of them
for idx, search_item in enumerate(all_search_items):
    result = hello_milvus.search(search_item, "embeddings", search_params, limit=1393, output_fields=["pk"])

    list_ = []
    for hits in result:
        for hit in hits:
            # print(f"hit: {hit}, pk field: {hit.entity.get('pk')}")
            # print("post and title", posts[hit.entity.get('pk')] + titles[hit.entity.get('pk')])
            list_.append({hit.entity.get('pk'): hit.distance})
    store_json[name_list[idx]] = list_
end_time = time.time()

with open("milve_res/doc_idx_distance_all.json", "w") as f:
    json.dump(store_json, f)

###############################################################################
# 7. drop collection
# Finally, drop the hello_milvus collection
print(fmt.format("Drop collection `hello_milvus`"))
utility.drop_collection("hello_milvus")
