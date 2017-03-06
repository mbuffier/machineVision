function r = testTimeGabor

% I load my previous results, the mixtures of gaussians and the prior for the apple images
load('RGBMixtureTrainingGabor.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
load('RGBTrainedGabor.mat', 'priorApple') ;

im1 = imread('../../testTimeApples/testApple1.jpg') ;
im2 = imread('../../testTimeApples/testApple2.jpg') ;
im3 = imread('../../testTimeApples/testApple3.jpg') ;

% here are the images I will fill with the rgb values and the results of my
% filters
img1 = zeros(12,size(im1,1)*size(im1,2)) ;
img2 = zeros(12,size(im2,1)*size(im2,2)) ;
img3 = zeros(12,size(im3,1)*size(im3,2)) ;

% I fill the rgb value
for i=1:3
    img1(i,:) = reshape(im1(:,:,i), 1,size(im1,1)*size(im1,2) ) ; % reshape the image 1
end
for i=1:3
    img2(i,:) = reshape(im2(:,:,i), 1,size(im2,1)*size(im2,2)) ; % reshape the image 2
end
for i=1:3
    img3(i,:) = reshape(im3(:,:,i), 1,size(im3,1)*size(im3,2) ) ; % reshape the image 3
end

% I fill the gabor result for the images
index = 4 ;
for theta= 1:59:119
    for sigm = 10:5:20
        gabor1 = gaborFilter(rgb2gray(im1),theta, sigm, 0.5) ;
        gabor1 = reshape(gabor1(:),1,size(gabor1,1)*size(gabor1,2)) ;
        img1(index,:) = gabor1 ;
        
        gabor2 = gaborFilter(rgb2gray(im2),theta, sigm, 0.5) ;
        gabor2 = reshape(gabor2(:),1,size(gabor2,1)*size(gabor2,2)) ;
        img2(index,:) = gabor2 ;
        
        gabor3 = gaborFilter(rgb2gray(im3),theta, sigm, 0.5) ;
        gabor3 = reshape(gabor3(:),1,size(gabor3,1)*size(gabor3,2)) ;
        img3(index,:) = gabor3 ;
        
        index = index+1 ;
    end 
end

% I compute my posteriors
posterior1 = uint8(255*real(getPosteriorProba(im1, img1,0.2 ,mixGaussEstApple, mixGaussEstNonApple))) ;
posterior2 = uint8(255*real(getPosteriorProba(im2, img2,0.2 ,mixGaussEstApple, mixGaussEstNonApple))) ;
posterior3 = uint8(255*real(getPosteriorProba(im3, img3,0.2 ,mixGaussEstApple, mixGaussEstNonApple))) ;

% load the groundTruth
groundTruth1 = rgb2gray(imread('../../testTimeApples/groundTruth1.png')) ;
groundTruth2 = rgb2gray(imread('../../testTimeApples/groundTruth2.png')) ;
groundTruth3 = rgb2gray(imread('../../testTimeApples/groundTruth3.jpg')) ;

% Compupte ROC curve for each image
[X1, Y1] = computeROCCurve(groundTruth1, posterior1) ;
[X2, Y2] = computeROCCurve(groundTruth2, posterior2) ;
[X3, Y3] = computeROCCurve(groundTruth3, posterior3) ;

% we find the best threshold for each image to compute quantitative results
thres1 = findThreshold(X1,Y1) ;
thres2 = findThreshold(X2,Y2) ;
thres3 = findThreshold(X3,Y3) ;

% we draw our results
figure('position', [0, 0, 1700, 800]) ;

subplot(3,4,1)
imshow(imread('../../testTimeApples/testApple1.jpg')) ;
subplot(3,4,2)
imshow(groundTruth1), title('Ground Truth of the first image') ;
subplot(3,4,3)
imshow(posterior1), title('Result from the algorithm') ;
subplot(3,4,4)
plot(X1,Y1, 'r-','LineWidth',1.2),
xlabel('False Positive rate') ,
ylabel('True Positive rate'),
title(['ROC curve, TPR = ' num2str(Y1(thres1)*100), '%, FPR = ' num2str(X1(thres1)*100) '%']) ;

subplot(3,4,5)
imshow(imread('../../testTimeApples/testApple2.jpg')) ;
subplot(3,4,6)
imshow(groundTruth2), title('Ground Truth of the second image') ;
subplot(3,4,7)
imshow(posterior2), title('Result from the algorithm') ;
subplot(3,4,8)
plot(X2,Y2, 'r-','LineWidth',1.2),
xlabel('False Positive rate') ,
ylabel('True Positive rate'),
title(['ROC curve, TPR = ' num2str(Y2(thres2)*100), '%, FPR = ' num2str(X2(thres2)*100) '%']) ;

subplot(3,4,9)
imshow(imread('../../testTimeApples/testApple3.jpg')) ;
subplot(3,4,10)
imshow(groundTruth3), title('Ground Truth of the third image') ;
subplot(3,4,11)
imshow(posterior3), title('Result from the algorithm') ;
subplot(3,4,12)
plot(X2,Y2, 'r-','LineWidth',1.2),
xlabel('False Positive rate') ,
ylabel('True Positive rate'),
title(['ROC curve, TPR = ' num2str(Y3(thres3)*100), '%, FPR = ' num2str(X3(thres3)*100) '%']) ;

saveas(gcf, 'questionGabor', 'jpg'); % save the result
end

% subroutine to compute the posterior probabillity knowing the prior and
% the mixture of gaussian
function posteriorApple = getPosteriorProba(im, img, priorApple, mixGaussEstApple,mixGaussEstNonApple )
[imY, imX,~] = size(im); %use of the original image im to have the original size of the image

priorNonApple = 1-priorApple ; % we compute the prior for a non apple pixel

likeApple = calcGaussianMixProb(img,mixGaussEstApple); % compute the probability to be an apple
likeNonApple = calcGaussianMixProb(img,mixGaussEstNonApple);% compute the probability not to be an apple

% compute posterior using bayes rules
posteriorApple = (likeApple.*priorApple)./(likeApple.*priorApple + likeNonApple.*priorNonApple);

% reshape the posterior
posteriorApple = reshape(posteriorApple, [imY, imX]);
end

%subroutine to return gaussian probabilities for a multidimensional X with
%several data points
function prob = getGaussProb(x,normalMean,normCov)
[nDim, nData] = size(x);
x = x-repmat(normalMean,1,nData);
prob = diag((x.')*(normCov\speye(size(normCov)))*x) ;
prob = (1/sqrt(det(normCov)*((2*pi)^nDim)))*exp(-0.5*prob);
prob = reshape(prob, [1, nData]) ;
end

% subroutine to return probability from a mixture of gaussian
function prob = calcGaussianMixProb(x,mixGaussEst)
k = mixGaussEst.k ;
% for each gaussian in the mixture we compute the probability of a pixel
% to be a part of it using the getGaussProb routine . We weight this probability by the weight of the
% gaussian in the mixture

%To optimize, I compute the probability on 3*500 submatrixes every time
[nDim, nData] = size(x);
optimisation = 500 ; % I choose the size of my submatrix to accelerate the computation

if(mod(nData,optimisation)~=0)
    xOptm = x(:,1:(optimisation*floor(nData/optimisation)+1)) ; % I reshape my data matrix
    nDataOpm = optimisation*floor(nData/optimisation)+1 ;
else
    xOptm = x(:,1:optimisation*floor(nData/optimisation)) ;
    nDataOpm = nData ;
end

prob = zeros(k,nData);

for i = 1:k % loop on gaussian
    % loop on data points
    for cData = 1:optimisation:(nDataOpm-optimisation) % instead of working on every data point, I work on "opmisation" data point
        prob(i,cData:cData+optimisation) = mixGaussEst.weight(1,i).*getGaussProb(xOptm(:,cData:cData+optimisation), mixGaussEst.mean(:,i), mixGaussEst.cov(:,:,i)) ;
    end
    % to avoid missing data point if the size of the image is not a
    % multiple of optimisation
    for cData = nData-optimisation:nData
        prob(i,cData) = mixGaussEst.weight(1,i).*getGaussProb(x(:,cData), mixGaussEst.mean(:,i), mixGaussEst.cov(:,:,i)) ;
    end
    prob = sum(prob,1) ;
end
end

% algorithm to get an threshold image
function B = threshold(img, k)
% img is the original image
% k is the threshold we choose to apply
% B is the binary result of the segmentation
[M, N] = size(img) ;
B = zeros(M,N, 'uint8') ;
for i=1:M
    for j=1:N
        if(img(i,j) >= k) % if the pixel intensity value is above the threshold, we set his value to 255
            B(i,j) = 255 ;
        else % if not, the pixel will be black
            B(i,j) = 0;
        end
    end
end
end

% using that function we can know which number of gaussian produces the
% best result. Return the threshold used to compute the FPR and TPR given
% the best distance to (0,1)
function thresValue = findThreshold(X, Y)
sizeX = size(X,2) ;
A = ones(1,sizeX) ;
dist = sqrt(X.^2+(Y-A).^2);
minDiff = find(dist==min(dist)) ;
thresValue = minDiff(1) ;
end

% this routine allows to compute the ROC curve for a given posterior and a
% groundtruth
function [X, Y] = computeROCCurve(groundTruth, posteriorApple)
[M, N] = size(posteriorApple) ;
tp = 0; % number of true positive
tn = 0 ; % number of true negative
fp = 0 ; % number of false positive
fn = 0 ; % number of false negative

numberOfThreshold = 255 ; %the number of threshold we will use to plot the ROC curve
X = zeros(1,numberOfThreshold) ; % contain the false positive rate
Y = zeros(1,numberOfThreshold) ; % contain the true positive rate

for i= 1:numberOfThreshold
    A = threshold(posteriorApple,i) ;
    for m=1:M
        for n=1:N
            if(groundTruth(m,n) == 0 && A(m,n) ==0)
                tn = tn+1 ;
            elseif(groundTruth(m,n) == 0 && A(m,n) == 255)
                fp = fp + 1 ;
            elseif(groundTruth(m,n) == 255 && A(m,n) == 0)
                fn = fn +1 ;
            elseif(groundTruth(m,n) == 255 && A(m,n) == 255)
                tp = tp +1 ;
            end
        end
    end
    X(1, i) = fp/(tn+fp) ; % false positive rate
    Y(1,i) = tp/(tp+fn) ; %true positive rate
    fp = 0;
    fn = 0 ;
    tp =0 ;
    tn = 0 ;
end
end


% function to perform the filtering in frenquency domain
function result = gaborFilter(img, angle,  sigma, lambda)

% img is the original image
% angle is the angle of the gobor filter
% sigma is the variance of the gaussian envelop
% lambda is the wavelength of the sinusoidal factor

[rows, cols] = size(img);
windowS = 15 ; %size of the filter window

% we create a meshgrid to use our filter
[X, Y] = meshgrid(-floor(windowS/2):floor(windowS/2) , -floor(windowS/2):floor(windowS/2)) ;

% we compute the rotation
X1 = X.*cos(angle)+Y.*sin(angle) ;
Y1 = -X.*sin(angle)+Y.*cos(angle) ;

% creation of the two part of the filter, the gaussian and the rotation
% part
gauss_part = 1/(2*pi*sigma^2)*exp(-((X1.^2+Y1.^2)./(2*sigma^2))) ;
rot_part = exp(-2*pi*1i*X1/lambda) ;

% put the image and the filter in the frequency domain with good dimension
gabor_fft = fft2(gauss_part*rot_part,rows,cols);
img_fft = fft2(img, size(gabor_fft, 1), size(gabor_fft, 2));

% Perform filtering.
result = real(ifft2(gabor_fft.*img_fft));
% Crop to original size.
result = result(1:size(img, 1), 1:size(img, 2));

% rescale the result image between 0 and 1
result = (result - min(result(:))) / (max(result(:)) - min(result(:))) ;

end