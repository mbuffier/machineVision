function practical2b

%The goal of this part of the practical is to take a real image containing
%a planar black square and figure out the transformation between the square
%and the camera.  We will then draw a wire-frame cube with it's base
%corners at the corner of the square.  You should use this
%template for your code and fill in the missing sections marked "TO DO"

%load in image 
im = imread('test104.jpg');

%define points on image
xImCart = [  140.3464  212.1129  346.3065  298.1344   247.9962;...
             308.9825  236.7646  255.4416  340.7335   281.5895];
         
%define 3D points of plane
XCart = [-50 -50  50  50 0 ;...
          50 -50 -50  50 0;...
           0   0   0   0 0];

%We assume that the intrinsic camera matrix K is known and has values
K = [640  0    320;...
     0    640  240;
     0    0    1];
       
%TO DO Use your routine to calculate TEst, the extrinsic matrix relating the
%plane position to the camera position.
TEst = estimatePlanePose(xImCart,XCart,K) ;

%define 3D points of plane
XWireFrameCart = [-50 -50  50  50 -50 -50  50  50;...
                   50 -50 -50  50  50 -50 -50  50;...
                    0   0   0   0 -100 -100 -100 -100];

%TO DO Draw a wire frame cube, by projecting the vertices of a 3D cube
%through the projective camera and drawing lines betweeen the resulting 2d image
%points
xImCartEst = projectiveCamera(K,TEst,XWireFrameCart) ;

%draw image and 2d points
figure; set(gcf,'Color',[1 1 1]);
imshow(im); axis off; axis image; hold on;
plot(xImCartEst(1,:),xImCartEst(2,:),'r.','MarkerSize',10);

for cPoint = 1:4
    %plot a green line between each pair of points
    pt2 = mod(cPoint,4)+1 ;
    pt3 = mod(cPoint+2,4)+1 ;
    plot([xImCartEst(1,cPoint) xImCartEst(1,pt2)],[xImCartEst(2,cPoint) xImCartEst(2,pt2)],'g-','LineWidth',2);
    hold on;
    plot([xImCartEst(1,cPoint+4) xImCartEst(1,pt2+4)],[xImCartEst(2,cPoint+4) xImCartEst(2,pt2+4)],'g-','LineWidth',2);
    hold on;
    plot([xImCartEst(1,cPoint) xImCartEst(1,cPoint+4)],[xImCartEst(2,cPoint) xImCartEst(2,cPoint+4)],'g-','LineWidth',2);
end;

saveas(gcf, 'question D.jpg') ;

%QUESTIONS TO THINK ABOUT...

%Do the results look realistic?
%If not, then what factors do you think might be causing this?
end

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