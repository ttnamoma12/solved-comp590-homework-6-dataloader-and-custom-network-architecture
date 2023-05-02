Download Link: https://assignmentchef.com/product/solved-comp590-homework-6-dataloader-and-custom-network-architecture
<br>
In this assignment you will practice with writing your own dataloader and custom network architecture and training Convolutional Neural Networks for image classification tasks.




<strong>Setup </strong>




You can work on the assignment on google <u>​</u><a href="https://colab.research.google.com/notebooks/intro.ipynb">colab</a>​.




<strong>Pytorch neural network tutorial </strong>

<a href="https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py">https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginn </a><a href="https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py">er-blitz-neural-networks-tutorial-py</a>

<h1>1.  Implement a pytorch dataset class to load training data.​</h1>

All the code in this task is in “Custom dataset cell block”. In block, you will implement a “CustomTrainDataset” class. The “CustomValDataset” is already written as an example for you, but don’t directly copy it because the data folder structure is different. Next, we will start with the basics.

To train a neural network, we need to load training samples sequentially in mini-batches. And this is one of the most important aspects of training neural networks. According to different data structures, different dataset classes are implemented to load data from them. The three necessary functions for a dataset class are the <strong>__init__(self, data_dir, transform=None)</strong>​    function, the <strong>__len__(self)</strong>​ function and the ​ <strong>__getitem__(self, idx)</strong>​ function. You may have​ noticed the “__” on both sides of the function name; these functions are the special private functions in python. For example, the __init__ function will be triggered when you instantiate a class: instance = MyClass(), it’s called implicitly, and whatever procedure you wrote in it will be executed. The __len__ function is triggered when you call len(instance), it will directly output the length you want to say. The __getitem__(self, idx) function has a necessary parameter idx; you should return an object according to that index. Now, you may have realized: you should already have a list of samples constructed in __init__, and __getitem__ will simply return the idx-th item of that list, subject to some transform.

In our application, we should construct a list of (image_path, class_id) tuples in __init__. The training data folder looks like this:

Cat/images/image001.jpeg

…

Cat/images/image500.jpeg

Dog/images/image001.jpeg

…

Dog/images/images500.jpeg

…

In the “train” folder of the tiny-imagenet-200 datasets, there are 200 folders, each one of them representing a class name. Inside each folder, there’s an image folder containing 500 images of that class. Therefore, your __init__(self, data_dir, transform=None) function will take a parameter called “data_dir” which is “prefix/tiny-imagenet-200/train”. It should read the filenames and structure in this folder, and merge all the information into a list of (image_path, class_id) pairs. This list should be self.samples. The order in the list doesn’t matter it because it will be shuffled outside of this class. Read the main block and the CustomValDataset class for what the

“transform” is. It’s basically a sequence of random data augmentation operations and conversion to pytorch tensor. Also, you can see where and how this dataset class is instantiated and used.

Pay attention to the class_id. You should first use <strong><u>sorted</u></strong><u>​ (</u><u>​ os.listdir(“data_dir”))</u> to get a sorted​ list of the class names, the class_id is the index of each class name in this list. You must sort it because we want the class_id to stay consistent with the validation dataset.

The __len__ function will trivially return the length of self.samples.

Now, given idx, in __getitem__(self, idx) you can get a pair (image_path, class_id) by self.samples[idx]. You now use the given pil_loader(image_path) function to read an image from image_path. Nextyou should apply self.transform to the image (see CustomValDataset for how to do that). Finally, you should return the pair of (transformed_image, class_id).

CustomValDataset is a good reference, but remember it has a different assumption about folder structure. Also, some code in CustomValDataset is not needed in your CustomTrainDataset because CustomValDataset was copied elsewhere. Keep your CustomTrainDataset as simple as possible. The parameters mentioned earlier is enough.

This dataset should have the same behavior as torchvision.datasets.ImageFolder, just for debugging. However, the latter one is made much more complicated to comply with more situations and we don’t need that. It’s not worth the time for you to adapt from there. This task is simple enough for you to implement from scratch. Simply Inheriting ImageFolder or its close parent classes is NOT acceptable.




<h1>2.  Implement a modified class of Alexnet</h1>

If you read the main block, you should have noticed the line model = CustomAlexnet(200)

This CustomAlexnet is what you are going to implement. Same as the loss class, this class is also a child class of nn.Module, and it only need two functions: <strong>__init__(self, num_class=200)</strong>​              and <strong>forward(self, x)</strong>​

In __init__(self, num_class=200), you will define the architecture using nn.Sequential, nn.Conv2d, nn.ReLU, nn.MaxPool2d and nn.Dropout.

In forward(self, x), x is an input tensor of shape (batch_size, 3, 64, 64), you need to use the structure defined in __init__ to process this x and return the output tensor, this output tensor should have size (batch_size, num_class).

You can refer to ​ <a href="https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py">pytorch’s Alexnet implementatio</a><u>​               </u><a href="https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py">n</a> for a quick start. Original Alexnet assumes​            input image size = 224, but tiny-imagenet-200 dataset has size 64. Original Alexnet assumes output size = 1000, but tiny-imagenet-200 dataset has 200 classes. To comply with this change and to strengthen your understanding, please implement the following changes:

<ul>

 <li>For the first nn.Conv2d layer, change the kernel_size to 8 and stride to 2. This change can reduce parameters and make the shapes of layers directly match.</li>

 <li>Remove the nn.AdaptiveAvgPool2d layer. This layer is unnecessary if the input size is fixed.</li>

 <li>Replace all the nn.Linear layers with nn.Conv2d layers with appropriate parameters to achieve the same purpose. After the final nn.Conv2d layer in our implementation, the output shape should be (batch_size, 200, 1, 1).</li>

 <li>In forward(self, x), use torch.flatten() function to convert the (batch_size, 200, 1,1) tensor to (batch_size, 200) tensor. This function was originally used between nn.Conv2d and nn.Linear layers, but since we decided to fully use nn.Conv2d, we will use torch.flatten() in the last step.</li>

 <li>Make sure the num_class = 200 instead of 1000 by passing an appropriate number.</li>

</ul>

A useful debugging command is the “summary(model, (3, 64, 64))” line in main block; it will automatically generate the output shape for each layer when the input images are 64×64:




As you can see, the output shape of MaxPool2d-13 matches the kernel size of Conv2d-15 (kernel_size=6). That’s why Conv2d-15 can directly convert feature maps into 1D vectors.







<ol start="3">

 <li>Run the training code for 100 epochs and get the best performing model​ on the validation set.</li>

</ol>

When you have implemented all the above classes, you should be able to train the network.




The training code will automatically save the best model to “model_best.pth.tar” depending on the top 1 accuracy on the validation set.


