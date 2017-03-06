function r = QuestionE
close all ;
clear all ;

% load the groundTruth
groundTruth1 = rgb2gray(imread('../../testTimeApples/groundTruth1.png')) ;
groundTruth2 = rgb2gray(imread('../../testTimeApples/groundTruth2.png')) ;
groundTruth3 = rgb2gray(imread('../../testTimeApples/groundTruth3.jpg')) ;

% load the posterior computed with question C
posteriorApple1 = imread('../../testTimeApples/postROC1.jpg') ;
posteriorApple2 = imread('../../testTimeApples/postROC2.jpg') ;
posteriorApple3 = imread('../../testTimeApples/postROC3.jpg') ;

% Compupte ROC curve for each image
[X1, Y1] = computeROCCurve(groundTruth1, posteriorApple1) ;
[X2, Y2] = computeROCCurve(groundTruth2, posteriorApple2) ;
[X3, Y3] = computeROCCurve(groundTruth3, posteriorApple3) ;

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
imshow(posteriorApple1), title('Result from the algorithm') ;
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
imshow(posteriorApple2), title('Result from the algorithm') ;
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
imshow(posteriorApple3), title('Result from the algorithm') ;
subplot(3,4,12)
plot(X2,Y2, 'r-','LineWidth',1.2),
xlabel('False Positive rate') ,
ylabel('True Positive rate'), 
title(['ROC curve, TPR = ' num2str(Y3(thres3)*100), '%, FPR = ' num2str(X3(thres3)*100) '%']) ;

saveas(gcf, 'questionE', 'jpg'); % save the result

end

% function to threshold our posterior result. If a pixel intensity is above
% the threshold, it's intensity is now 255, if it's bellow, its intensity
% is now 0. 
function B = threshold(img, k)
% img is the original image we want to segment
% k is the threshold we choose
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
B = uint8(B) ;
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