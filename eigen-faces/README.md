# eigen faces

## Introduction

Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition.The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set.
[Wikipedia](https://en.wikipedia.org/wiki/Eigenface)

I have tried to find eigenfaces for my fellow [batchmates](https://www.usfca.edu/arts-sciences/graduate-programs/data-science/our-students) in MSAN (USF) and my [professors](https://www.usfca.edu/arts-sciences/graduate-programs/data-science/faculty).

## External files required for face cropping and alignment
[haarcascade_frontalface_default.xml](http://gregblogs.com/computer-vision-cropping-faces-from-images-using-opencv2/)

[shape_predictor_68_face_landmarks.dat](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)


## Results

### Top 4 eigenfaces for students
![Eigenfaces for students](../imgs/eigenfaces-student.png)

### Top 4 eigenfaces for faculty
![Eigenfaces for faculty](../imgs/eigenfaces-faculty.png)

### Image reconstruction using eigenfaces
![Reconstruction for faculty](../imgs/reconstruct-faculty.png)

