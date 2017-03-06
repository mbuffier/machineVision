function r = QuestionC

close all;

%************* Code for QUESTION C *****************

 % I load my previous results, the mixtures of gaussians and the prior for the apple images
load('RGBMixtureTraining7.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
load('RGBTrained.mat', 'priorApple') ;

% The 3 images we would like to find the apple in 
im1 = double(imread('../../validationApples/Apples_by_MSR_MikeRyan_flickr.jpg'))/255 ;
im2 = double(imread('../../validationApples/audioworm-QKUJj2wmxuI-original.jpg'))/255 ;
im3 = double(imread('../../validationApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg'))/255 ;

% we compute the posterior probability using the routine getPosteriorProba
posteriorApple1 = getPosteriorProba(im1,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
posteriorApple2 = getPosteriorProba(im2,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
posteriorApple3 = getPosteriorProba(im3,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;

% We plot the orignal image, the posterior result and an threshold image
% using the threshold routine (First, I put arbitrarily a threshold to 0.5)
fig1 = figure ;
set(fig1, 'Position', [0 0 1000 800]) ;

subplot(3,3,1), imshow(im1);
subplot(3,3,2), imshow(posteriorApple1, [min(posteriorApple1(:)) max(posteriorApple1(:))]);
subplot(3,3,3), imshow(threshold(posteriorApple1, 0.5));
subplot(3,3,4), imshow(im2);
subplot(3,3,5), imshow(posteriorApple2,[min(posteriorApple2(:)) max(posteriorApple2(:))]);
subplot(3,3,6), imshow(threshold(posteriorApple2, 0.5));
subplot(3,3,7), imshow(im3);
subplot(3,3,8), imshow(posteriorApple3,[min(posteriorApple3(:)) max(posteriorApple3(:))]);
subplot(3,3,9), imshow(threshold(posteriorApple3, 0.5));

saveas(gcf, 'questionC', 'jpg'); % save the result


%************* Code for QUESTION D *****************


% Those posteriors images will be used in questionD to compute the ROC
% curves 
load('RGBTrained.mat', 'priorApple') ;
im3 = double(imread('../../validationApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg'))/255 ;

load('RGBMixtureTraining3.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
postROC3 = getPosteriorProba(im3,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC3,'../../validationApples/postROC3.jpg') ;

load('RGBMixtureTraining5.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
postROC5 = getPosteriorProba(im3,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC5,'../../validationApples/postROC5.jpg') ;

load('RGBMixtureTraining7.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
postROC7 = getPosteriorProba(im3,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC7,'../../validationApples/postROC7.jpg') ;

load('RGBMixtureTraining10.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
postROC10 = getPosteriorProba(im3,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC10,'../../validationApples/postROC10.jpg') ;


%************* Code for QUESTION E *****************


%Those results will be used in Question E during test time

load('RGBMixtureTraining3.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
load('RGBTrained.mat', 'priorApple') ;
im1 = double(imread('../../testTimeApples/testApple1.jpg'))/255 ;
im2 = double(imread('../../testTimeApples/testApple2.jpg'))/255 ;
im3 = double(imread('../../testTimeApples/testApple3.jpg'))/255 ;

postROC1 = getPosteriorProba(im1,0.2 ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC1,'../../testTimeApples/postROC1.jpg') ;

postROC2 = getPosteriorProba(im2,0.2 ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC2,'../../testTimeApples/postROC2.jpg') ;

postROC3 = getPosteriorProba(im3,priorApple ,mixGaussEstApple, mixGaussEstNonApple) ;
imwrite(postROC3,'../../testTimeApples/postROC3.jpg') ;

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
prob = diag((x.')*inv(normCov)*x) ;
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
