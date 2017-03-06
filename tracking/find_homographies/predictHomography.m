function r=practical2

%This project explores the geometry of a single camera. The aim is to take several points on
%a plane, and predict where they will appear in the camera image. Based on these observed
%points, we will then try to re-estimate the Euclidean transformation relating the plane and
%the camera. In practical 2b we will use this code to draw a wireframe cube
%on an augmented reality marker.   You should use this
%template for your code and fill in the missing sections marked "TO DO"


%We assume that the intrinsic camera matrix K is known and has values
K = [640  0    320;...
     0    640  240;
     0    0    1];
 
%We will assume an object co-ordinate system with the Z-axis pointing upwards and the
%origin in the centre of the plane. There are four known points on the plane, with coordinates
%(in mm):

XCart = [-100 -100  100  100 0 ;...
         -100  100  100 -100 0;...
          0    0    0    0   0 ];

%We will assume that the correct transformation from the plane co-ordinate system to the
%camera co-ordinate system (extrinsic matrix) is:

T = [ 0.9851  -0.0492  0.1619  46.00;...
     -0.1623  -0.5520  0.8181  70.00;...
      0.0490  -0.8324 -0.5518  500.89;...
      0        0       0       1] ;
  
% TO DO  Use the general pin-hole projective camera model discussed in the lectures to estimate 
%where the four points on the plane will appear in the image.  Fill in the
%details of the function "projectiveCamera" - body of function appears below

xImCart = projectiveCamera(K,T,XCart);

% TO DO Add noise to the pixel positions to simulate having to find these points in a noisy
%image. Store the results back in xImCart.  
%The noise should have standard deviation of one pixel in each direction.

xImCart = xImCart + randn(size(xImCart)) ;

%Now we will take the image points and the known positions on the card and try to
%estimate the extrinsic matrix using the algorithm discussed in the lecture. 
%Fill in the details of the function "estimate plane pose" - body of function appears
%below

TEst = estimatePlanePose(xImCart,XCart,K) ;

sqDiff = mean(sum((T-TEst).^2)) ;

disp(['The square difference between T and TEst is : ' num2str(sqDiff)]);

end

%if you have got this correct, it should resemble T above.

%==========================================================================
%==========================================================================

%goal of function is to project points in XCart through projective camera
%defined by intrinsic matrix K and extrinsic matrix T.
function xImCart = projectiveCamera(K,T,XCart)

%TO DO convert Cartesian 3d points XCart to homogeneous coordinates XHom
XHom = [XCart; ones(1,size(XCart,2))] ;

%TO DO apply extrinsic matrix to XHom to move to frame of reference of
%camera
XHom = T*XHom ;

%TO DO project points into normalized camera coordinates xCamHom by (achieved by
%removing fourth row)
xCamHom =  XHom(1:3,:) ;

%TO DO move points to image coordinates xImHom by applying intrinsic matrix
xImHom = K*xCamHom ;

%TO DO convert points back to Cartesian coordinates xImCart
xImCart = xImHom(:,:)./repmat(xImHom(3,:),3,1) ;

xImCart = xImCart(1:2, :);
end

%==========================================================================
%==========================================================================

%goal of function is to estimate pose of plane relative to camera
%(extrinsic matrix) given points in image xImCart, points in world XCart
%and intrinsic matrix K.

function T = estimatePlanePose(xImCart,XCart,K)


%TO DO Convert Cartesian image points xImCart to homogeneous representation
xImHom = [xImCart; ones(1,size(xImCart,2))] ; 

%TO DO Convert image co-ordinates xImHom to normalized camera coordinates
xCamHom = K^-1*xImHom ;

%TO DO Estimate homography H mapping homogeneous (x,y)
%coordinates of positions in real world to xCamHom.  Use the routine you wrote for
%Practical 1B.

H = calcBestHomography(XCart, xCamHom(1:2,:)) ;

%TO DO Estimate first two columns of rotation matrix R from the first two
%columns of H using the SVD

HTest = H(:,1:2) ;

[U,~,V] = svd(HTest) ;
inter = [ 1 0 ; 0 1; 0 0] ;
rotation = U*inter*(V.') ;

%TO DO Estimate the third column of the rotation matrix by taking the cross
%product of the first two columns

lastC = cross(rotation(:,1),rotation(:,2)) ;

%TO DO Check that the determinant of the rotation matrix is positive - if
%not then multiply last column by -1.

rotation = [rotation lastC] ;
deter = det(rotation);
if (deter == -1)
    rotation(:,3) = -1*rotation(:,3) ;
end

%TO DO Estimate the translation t by finding the appropriate scaling factor k
%and applying it to the third colulmn of H
inter = H(:,1:2)./rotation(:,1:2) ;
lambda = (1/6)*sum(inter(:)) ;

taux = H(:,3)/lambda ;

%TO DO Check whether t_z is negative - if it is then multiply t by -1 and
%the first two columns of R by -1.

if (taux(3,1) < 0 )
    rotation(:,1:2) = -1*rotation(:,1:2) ;
    taux = -1*taux ;
end

T = [rotation taux; 0 0 0 1] ;

end

function H = calcBestHomography(pts1Cart, pts2Cart)

%should apply direct linear transform (DLT) algorithm to calculate best
%homography that maps the points in pts1Cart to their corresonding matchin in 
%pts2Cart

%****TO DO ****: replace this
M = size(pts1Cart,2) ;

%then construct A matrix which should be (10 x 9) in size
%solve Ah = 0 by calling h = solveAXEqualsZero(A); (you have to write this routine too - see below)
A = zeros(2*M,9);
index = 1 ;

for i = 1:M
    test = [0, 0,0, -pts1Cart(1,i), -pts1Cart(2,i), -1, pts2Cart(2,i).*pts1Cart(1,i), pts2Cart(2,i).*pts1Cart(2,i), pts2Cart(2,i)] ;
    A(index,:) = test ;
    A(index+1,:) = [pts1Cart(1,i) pts1Cart(2,i) 1 0 0 0 -pts2Cart(1,i)*pts1Cart(1,i) -pts2Cart(1,i)*pts1Cart(2,i) -pts2Cart(1,i)] ;
    index = index+2 ;
end

h = solveAXEqualsZero(A); 
%reshape h into the matrix H

%Beware - when you reshape the (9x1) vector x to the (3x3) shape of a homography, you must make
%sure that it is reshaped with the values going first into the rows.  This
%is not the way that the matlab command reshape works - it goes columns
%first.  In order to resolve this, you can reshape and then take the
%transpose

H = reshape(h, [3,3]) ;
H = H.' ;
end

%==========================================================================
function x = solveAXEqualsZero(A)
[~,~,V] = svd(A) ;
N = size(V,2) ;
x = V(:,N) ;
end

