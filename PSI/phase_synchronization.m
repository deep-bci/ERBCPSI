function [PSI,PLI,wPLI]=phase_synchronization(X,Y)
%computes the phase synchronization index based on the Hilbert transform
%
%PSI=phase_synchronization(X,Y)
%
%computes the phase synchronization indices (PSI) for functional 
%connectivity studies between two data vectors X and Y or between all rows 
%of a data matrix X. The instantaneous phases are determined based on the 
%Hilbert transform.

%
%INPUT: X - N x T  data matrix (N: # of sensors, T: # of time samples) or
%           first data vector
%       if X is a vector:
%       Y - second data vector
%
%OUTPUT: PSI - if X is a matrix: N x N matrix containing the PSI values 
%                                between all channel pairs
%                               
%              if X and Y are vectors: scalar corresponding to the PSI
%                                      value between signals X and Y
%                                 


if nargin<2
    %working on matrix X (channels x time)
    PSI=eye(size(X,1));
    PLI=eye(size(X,1));
    wPLI=eye(size(X,1));
    band_hilbert = hilbert(X');
    phi_x=angle(band_hilbert);
    
%     perc10w =  floor(size(phi_x,1)*0.1);
%     phi_x = phi_x(perc10w+1:end-perc10w,:);
%     band_hilbert = band_hilbert(perc10w+1:end-perc10w,:);
    
    for k1=1:size(X,1)
        for k2=k1+1:size(X,1)
            phi=phi_x(:,k1)-phi_x(:,k2);
            PSI(k1,k2)=abs(mean(exp(1i*phi)));
            PSI(k2,k1)=PSI(k1,k2);
            
%              PLI(k1,k2) = abs(mean(sign((abs(phi)- pi).*phi)));
%              PLI(k2,k1)=PLI(k1,k2);
             
%              crossspec = band_hilbert(:,k1).* conj(band_hilbert(:,k2)); 
%              crossspec_imag = imag(crossspec); 
%              wPLI(k1,k2) = abs(mean(crossspec_imag))/mean(abs(crossspec_imag)); 
%              wPLI(k2,k1)=wPLI(k1,k2);
        end
    end
else
    %working on two vectors X and Y
    phi_x=angle(hilbert(X));
    phi_y=angle(hilbert(Y));
    phi=phi_x-phi_y;
    PSI=abs(mean(exp(1i*phi)));
end