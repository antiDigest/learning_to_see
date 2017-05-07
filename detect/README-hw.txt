# README.txt - Tensorflow Homework 10

All of these programs have been run and tested on a MAC OS X.

---------------------------------------------------------INSTALLATION------------------------------------------------------------

The program requires installation of python3 on MAC OS and installing tensorflow.

Tensorflow with python3 can simply be installed by running:


		"pip3 install tensorflow"

----------------------------------------------------------HOW TO RUN-------------------------------------------------------------

python3 running the program is simple:


		"python3 proj3.py"


---------------------------------------------------------DOCUMENTATION-----------------------------------------------------------

EXPERIMENTS:

The program was run using various methods of implementation with Convolutional Neural Networks. The initial code is first running the max pool 2*2 layer and then Convolutional Layer which is running a 3*3 conv layer (we're referring to this as conv1 everywhere) over the max pooled image. This is then converted to a fully connected layer of 1024 elements which is finally connected to an output layer.

I will be mentioning the changes I did to the code and how it improved the prediction accuracy. With all of these changes, I do also mention the changes made with the learning rate and the batch size and how they are affecting the accuracy.


CHANGE #1 : ADDED AN EXTRA CONV LAYER OF 5*5 WINDOW SIZE (KEEPING THE OUTPUT OF SAME SHAPE):

We'll be referring to this layer as conv2 everywhere. Just adding an extra conv layer does not always improve the accuracy as we shall see but as an extra conv layer is added, the accuracy improves. This is probably because the neural network gets to extract more features and learn in more detail about the image which in turn helps in improving the predictions. The accuracy goes up to 0.9219.

With this changes in the batch size only affect the accuracy when the batch size is increased to 5 times the initial (50), and the change is approximately 0.002 which does not help a lot.


CHANGE #2 : ADDED AN EXTRA CONV LAYER OF 3*3 WINDOW SIZE (KEEPING THE OUTPUT OF VALID SHAPE):

Like I mentioned in the previous change discussion, adding this layer did not help too much, there was not much change in accuracy (the accuracy stayed around 0.92), and so just keeping on adding conv layers will not help with improving the accuracy. Even the batch size and learning rate changes do not affect a lot with this.


CHANGE #3: REMOVED THE CONV LAYER OF 3*3 WINDOW SIZE AND ADDED A MAX POOL LAYER BETWEEN THE CONV1 AND CONV2:

Adding a max pool layer always helps with improving the accuracy. The pool layers always reduce the size of the image (even when you put in 'SAME' in the padding parameter) so the next layer need to be kept in flow with the hyperparameters given. The pool layer actually improves the accuracy. This means that between two conv layers, a pool layer helps. The accuracy went up to about 0.9356.


CHANGE #4: REPLACED CONV2 BY TWO CONV LAYERS OF 3*3 WINDOW FOLLOWED BY 1*1 WINDOW -- COMBINATION IS CONV2:

This replacement did not change anything, which actually means that the 5*5 window is exactly equal to a 3*3 window followed by a 1*1 window. A 1*1 window also works as it is performing dot product over the depth of the image, which I have set to 128 in this particular conv layer. As you can also notice, the number of weights in a 3*3*128 weight vector = 1152 and number of weights in 1*1*128 weight vector = 128 which makes the sum of number of the new conv2 weights to be 1280. If you calculate the number of weights in a 5*5*128 weight vector which comes out to be 3200. Hence we are actually reducing the space requirements of a 5*5*128 weight vector by converting it into a 3*3*128 weight followed by a 1*1*128 weight vector, and this also runs faster than the previous configuration due to reduced weights to optimize during the learning phase. The fast computation is not exactly something which is a considerable change.

CHANGE #5: ADDED SOME MORE CONV LAYERS AND SOME CONV LAYERS WORKING IN PARALLEL -- COMBINATION IS CONV3:

The conv3 did not drastically change the accuracy, but it was an improvement of some 1% in the accuracy which was a motivation to add more layers and you can see a conv4 in the code as well. Since the size of the images is too small, adding a pool layer in between only reduces the size of the image to 1*1 which can only run a 1*1 weight vector, so no more pool layers were added. The final accuracy achieved with this model was 0.9429.

Conv layers working in parallel only means that both take input from conv2 and produce an output which is then summed up and finally becomes an input to conv4.


CHANGE #6: INCREASED BATCH SIZE TO 500:

Increasing the batch size to 500 was actually a wrong decision, it started taking too long for me to be patient and wait for the final test accuracy, hence the code was stopped while it was running. A better working solution was batch size of 250. Batch sizes of 250 improved the accuracy with this new model of 4 conv layers to 0.9526.


CHANGE #7: DECREASED LEARNING RATE TO 0.001:

Decreasing learning rate slowed down the process and an accuracy of 0.9371 was achieved in doing so. The learning rate put back to 0.005 was better. Even though learning rate changes do not slow down the learning time of 1000 iterations, it slows down the learning rate, which means that to reach the same accuracy as the previous setup, it needs to run more iterations.


CHANGE #8: ADAM OPTIMISER:

The adam optimizer increases the learning rate using the concept of momentum. As considered an optimal optimiser, my final change reflected an accuracy increase from 0.9579 to 0.9744, which is an improvement of approximately 2%. It does not affect the time a lot, though there was an improvement of around 3-4 minutes of learning. The adam optimiser was faster than the Gradient Descent optimiser and produced an improvement of 2% at least. Even with changes such as decreasing the batch size to 150, the improvement in time as well as accuracy was significant.


CHANGE #9: LEARNING RATE TO 0.007:

The learning works best on a learning rate of 0.007. Increasing learning rate more than 0.007 actually reduces the final test accuracy.


The final code submitted has a test accuracy of 0.9778 which has been the highest accuracy with all the changes.

