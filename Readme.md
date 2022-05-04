# SecureBiNN

<p align='left'>
SecureBiNN is a novel 3-party secure computation framework1for evaluating privacy-preserving discrete 
neural network (DiNN). In SecureBiNN, input data and 
model parameters are separated uniformly at random among the parties who may then act upon secure computations to obtain secret shares of the prediction result without disclosing the input data, model parameters and the prediction result. SecureBiNN performs linear operations in a computation-efficient and communication-free way (by 3-party secret sharing). For non-linear operations, we provide novel methods to evaluate maxpooling layers and batch normalization8layers in DiNN and the communication cost is significantly minimized comparing to previous work like ABY3(CCS’18) and SecureNN (PETS’19). SecureBiNN10conforms to universally composable (UC) security and can be implemented by11tensorflow and pytorch. We evaluate an illustrative instance on VGG16 within 120.986 seconds (2×faster than XONN) and 1.01MB communication cost (13×less than Falcon).
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
