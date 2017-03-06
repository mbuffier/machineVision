function r = testTime

% I load my previous results, the mixtures of gaussians and the prior for the apple images
load('RGBMixtureTrainingHSV.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
load('RGBTrainedHSV.mat', 'priorApple') ;

im1 = rgb2hsv(imread('../../testTimeApples/testApple1.jpg')) ;
im2 = rgb2hsv(imread('../../testTimeApples/testApple2.jpg')) ;
im3 = rgb2hsv(imread('../../testTimeApples/testApple3.jpg')) ;

posterior1 = uint8(255*real(getPosteriorProba(im1,0.2 ,mixGaussEstApple, mixGaussEstNonApple))) ;
posterior2 = uint8(255*real(getPosteriorProba(im2,0.2 ,mixGaussEstApple, mixGaussEstNonApple))) ;
posterior3 = uint8(255*real(getPosteriorProba(im3,0.2 ,mixGaussEstApple, mixGaussEstNonApple))) ;

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

saveas(gcf, 'questionHSV', 'jpg'); % save the result
end

% subroutine to compute the posterior probabillity knowing the prior and
% the mixture of gaussian
function posteriorApple = getPosteriorProba(img, priorApple, mixGaussEstApple,mixGaussEstNonApple )
[imY, imX,~] = size(img);

priorNonApple = 1-priorApple ; % we compute the prior for a non apple pixel
imgNew = zeros(3,imY*imX);

imgNew(1,:) = reshape(img(:,:,1), 1,imY*imX ) ; % reshape the image 
imgNew(2,:) = reshape(img(:,:,2), 1,imY*imX ) ;
imgNew(3,:) = reshape(img(:,:,3), 1,imY*imX ) ;

likeApple = calcGaussianMixProb(imgNew,mixGaussEstApple); % compute the probability to be an apple
likeNonApple = calcGaussianMixProb(imgNew,mixGaussEstNonApple);% compute the probability not to be an apple

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
