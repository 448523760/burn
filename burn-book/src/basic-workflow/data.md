# Data

Typically, one trains a model on some dataset. Burn provides a library of very useful dataset
sources and transformations. In particular, there are Hugging Face dataset utilities that allow to
download and store data from Hugging Face into an SQLite database for extremely efficient data
streaming and storage. For this guide, we will use the MNIST dataset provided by Hugging Face.

To iterate over a dataset efficiently, we will define a struct which will implement the `Batcher`
trait. The goal of a batcher is to map individual dataset items into a batched tensor that can be
used as input to our previously defined model.

```rust , ignore
use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

```

This codeblock defines a batcher struct with the device in which the tensor should be sent before
being passed to the model. Note that the device is an associative type of the `Backend` trait since
not all backends expose the same devices. As an example, the Libtorch-based backend exposes
`Cuda(gpu_index)`, `Cpu`, `Vulkan` and `Metal` devices, while the ndarray backend only exposes the
`Cpu` device.

这段代码定义了一个批处理器（batcher）结构体，用于将张量发送到模型之前指定的设备上。
注意，设备是Backend trait的关联类型，因为不同的后端可能不会暴露相同的设备类型。
例如，基于Libtorch的后端暴露了Cuda(gpu_index)、Cpu、Vulkan和Metal设备，而ndarray后端只暴露了Cpu设备。

Next, we need to actually implement the batching logic.

```rust , ignore
#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MNISTBatch { images, targets }
    }
}
```

In the previous example, we implement the `Batcher` trait with a list of `MNISTItem` as input and a
single `MNISTBatch` as output. The batch contains the images in the form of a 3D tensor, along with
a targets tensor that contains the indexes of the correct digit class. The first step is to parse
the image array into a `Data` struct. Burn provides the `Data` struct to encapsulate tensor storage
information without being specific for a backend. When creating a tensor from data, we often need to
convert the data precision to the current backend in use. This can be done with the `.convert()`
method. While importing the `burn::tensor::ElementConversion` trait, you can call `.elem()` on a
specific number to convert it to the current backend element type in use.


```text
MNISTBatch 结构体包含两个字段：images 和 targets，分别表示图像和对应目标的张量。

在实现中，batch 函数接受一个 MNISTItem 实例的向量作为输入，并返回一个 MNISTBatch 实例。
它处理输入项以创建 images 和 targets 张在提供的代码片段中，定义了一个名为 MNISTBatch<B: Backend> 的结构体。
它实现了 Batcher<MNISTItem, MNISTBatch<B>> trait，其中 B 表示所使用的后端（backend）。
MNISTBatch 结构体包含两个字段：images 和 targets，分别表示图像和对应目标的张量。

images 张量的创建涉及对每个项的图像数据执行一系列操作：
1. 将图像数据转换为 Data<f32, 2> 对象。
2. 使用适当的后端特定转换方法从数据创建张量。
3. 将张量重塑为维度为 [1, 28, 28]。
4. 将张量的值归一化为 [0, 1]，并具有均值为 0.1307 和标准差为 0.3081，这是根据 PyTorch MNIST 示例中的规范进行的。
   
targets 张量的创建涉及将每个项的标签转换为 i64，然后从中创建张量。
最后，images 和 targets 张量在第 0 维上进行连接，并移动到 self.device 指定的设备上。
使用这些张量创建一个 MNISTBatch 实例，并由 batch 函数返回。
```

