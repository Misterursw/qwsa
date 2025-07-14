from datasets import load_dataset

from datasets import Dataset

# 直接从单个 .arrow 文件加载
dataset = Dataset.from_file("/home/micintern/.cache/huggingface/datasets/Senqiao___lisa_plus_caption/default/0.0.0/5ea1b93d83dcbe0b7252b9ed5c355022ef94fd0d/lisa_plus_caption-train.arrow")

# 查看 dataset 的基本信息
#print(dataset)               # 打印出样本总数、列名等
print("列名：", dataset.column_names)

# 查看第一条样本
#print("第一条：", dataset[0])
record = dataset[0]
# print("English Question:", record["English Question"])
# print("English Answer:  ", record["English Answer"])
# print("Img Path:      ", record["img_path"])
# print("ID:      ", record["ID"])
# #print("points:      ", record["points"])
print("Chinese Question:", record["Chinese Question"])
print("English Question:", record["English Question"])
print("Chinese Answer:      ", record["Chinese Answer"])
print("English Answer:      ", record["English Answer"])
print("image_path:      ", record["image_path"])