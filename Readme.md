# SecureBiNN

<p align='left'>
	The paper proposes SecureBiNN, a novel three-party secure computation framework for evaluating privacy-preserving binarized neural network (BiNN) in semi-honest adversary setting.
In SecureBiNN, three participants hold input data and model parameters in secret sharing form, and execute secure computations to obtain secret shares of prediction result without disclosing their input data, model parameters and the prediction result.
SecureBiNN performs linear operations in a computation-efficient and communication-free way, and for non-linear operations, we provide novel secure methods for evaluating activation function, maxpooling layers, and batch normalization layers in BiNN.
Communication overhead is significantly minimized comparing to previous work like XONN and Falcon.
We implement SecureBiNN with tensorflow and the experiments show that on CIFAR-10 dataset, SecureBiNN achieves an accuracy of 81.5\% within 1.301 seconds and 16.60MB communication cost with the Fitnet structure.
Other concrete comparisons with state-of-the-art are presented as well.

</p>

## Illustration of this project
In theory, SecureBiNN supports arbitrary division of data and models among the three participants. However, for simplicity, 
we suppose three participants are data owner, model owner and 
a trusted third party. The data owner seperate the data to other participants, and the model owner seperates the shares of the model.

## Requirments
python == 3.8 <br>
Tensorflow == 2.4.1 <br>
numpy == 1.19.0<br>


## How to run this project ?
First fill in the relevant settings in role/config.json according to the actual situation, including the ips and the ports of three participants. Then, run<br>
<code>
python make_roles.py
</code>
<br>
to generate three files representing three different participants. For each participant, execute
<br>
<code>
python main.py
</code>
<br>
to run the SecureBiNN.
The report message will be returned to data owner.
