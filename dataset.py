import numpy as np
import mindspore.dataset as ds

class MyDataset:

    def __init__(self,data,label):
        
        self.data = data.astype(np.float32)
        self.label = label.astype(np.int32)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def create_dataset(data,label,params,is_batch=True):
    dataset_generator = MyDataset(data, label)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    if is_batch:
        dataset = dataset.batch(params.batch_size)
    else:
        dataset = dataset.batch(data.shape[0]) #take the whole data as one batch
    return dataset
    

if __name__=="__main__":
    # 实例化数据集类
    # 迭代访问数据集
    x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(params)
    dataset=create_dataset(x_train,y_train)
    for data,label in dataset:
        data1 = data['data'].asnumpy()
        label1 = data['label'].asnumpy()
        print(f'data:[{data1[0]:7.5f}, {data1[1]:7.5f}], label:[{label1[0]:7.5f}]')

    # 打印数据条数
    print("data size:", dataset.get_dataset_size())
