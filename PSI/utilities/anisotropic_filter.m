function anisotropic_filter (IM)
%inspired by http://www.researchgate.net/publication/221401076_Muliscale_Vessel_Enhancement_Filtering
%(Another option would be to
%do this separately with Vaa-3D, which has already implemented the entire pipeline 


%Steps for anisotropic filtering of an image : 

%figure out the appropriate scale over which to compute gradients (the
%scale should be roughly the size of local dendrites). Vaa-3D uses the
%local gray-weighted distance transform, though this requires labelling of background pixels

%compute discrete gradient and hessian at the desired scale using
%convolution with derivative of gaussian
%or convolution with gaussian, followed by subtraction (same)

%consider using toolboxes (downloaded, on dropbox) for gradient estimation

%get eigenvalues of hessian matrix

%apply contrast according to eigenvalues per most recent Peng paper on contrast enhancement



%derivative of gaussian
 dx=floor(3*sigmax);
 dy=floor(3*sigmay);
 [X,Y]=meshgrid(-dx:dx,-dy:dy);
 k=1/(sqrt(2*pi*sigmax*sigmay));
 f=k*exp(-0.5*((X.^2/((sigmax^2)))+(Y.^2/((sigmay^2)))));
 gauss_x=(-X./(sigmax^2)).*f;
 gauss_y=(-Y./(sigmay^2)).*f; 
 gauss_xy=(-X./(sigmax^2)).*gauss_y;
 gauss_xx=gauss_x.*(-X./(sigmax^2))+(-f/(sigmax^2));
 gauss_yy=gauss_y.*(-Y./(sigmay^2))+(-f/(sigmay^2)); 