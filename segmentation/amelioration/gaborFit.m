function r = gaborFit

% load the first mixture to fit the data 
load('RGBTrainedGabor.mat','RGBAppleFinal','RGBNonAppleFinal');

% I use 3 gaussians directly this time 
nGaussEst = 3 ;
% fit mixture of Gaussian model for apple data
mixGaussEstApple = fitMixGauss(RGBAppleFinal,nGaussEst);
% fit Mixture of Gaussian model for non-apple data
mixGaussEstNonApple = fitMixGauss(RGBNonAppleFinal,nGaussEst);
save('RGBMixtureTrainingGabor.mat', 'mixGaussEstApple', 'mixGaussEstNonApple') ;
end


function mixGaussEst = fitMixGauss(data,k)

[nDim nData] = size(data);
optimisation = 500 ; % I choose the size of my submatrix to accelerate the computation
data = data(:,1:(optimisation*floor(nData/optimisation)+1)) ; % I reshape my data matrix to avoid having indices above the matrix size
% I might lose between 1 and 500 data points, but knowing the number of
% total data points, I assume it was not very important
nData = optimisation*floor(nData/optimisation)+1 ;

postHidden = zeros(k, nData);

mixGaussEst.d = nDim;
mixGaussEst.k = k;
mixGaussEst.weight = (1/k)*ones(1,k);
mixGaussEst.mean = 2*randn(nDim,k); 
for cGauss =1:k
    mixGaussEst.cov(:,:,cGauss) = (1+1.5*rand(1))*eye(nDim,nDim);
end;

nIter = 20;
for cIter = 1:nIter
    %Expectation step
    for cData = 1:optimisation:(nData-optimisation) % instead of working on every data point, I work on "opmisation" data points
        for i=1:k
            % I compute the responsability for each point in my submatrix
            % and they are stored in the likeI matrix
            likeI = getGaussProb(data(:,cData:cData+optimisation), mixGaussEst.mean(:,i), mixGaussEst.cov(:,:,i)) ;
            postHidden(i,cData:cData+optimisation) = mixGaussEst.weight(1,i).*likeI ; 
        end
        postHidden(:,cData:cData+optimisation) = postHidden(:,cData:cData+optimisation)./repmat(sum(postHidden(:,cData:cData+optimisation),1),k,1) ;
    end;
    
    % maximization step 
    for cGauss = 1:k
        % update the weight
        mixGaussEst.weight(1,cGauss) = sum(postHidden(cGauss,:))/sum(sum(postHidden)) ;
        mixGaussEst.weight(1,cGauss) =real(mixGaussEst.weight(1,cGauss)) ;
        % update the mean vector
        postHiddenArray = repmat(postHidden(cGauss,:),nDim,1) ;
        mixGaussEst.mean(:,cGauss) =  sum(postHiddenArray.*data,2)./sum(postHidden(cGauss,:)) ;
        mixGaussEst.mean(:,cGauss) = real(mixGaussEst.mean(:,cGauss));
        % update the covariance matrix for each data 
        data2 = data - repmat(mixGaussEst.mean(:,cGauss),1,nData);
        mixGaussEst.cov(:,:,cGauss) = ((repmat(postHidden(cGauss,:), nDim,1).*data2)*data2.')./sum(postHidden(cGauss,:)) ;
        mixGaussEst.cov(:,:,cGauss) = real(mixGaussEst.cov(:,:,cGauss)) ;
    end;    
end;
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
