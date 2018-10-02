function [kmin,kmax,vmin,vmax, A, B, C, cflag] = curv(nflag,x,y,z,nrm,nb)

% curv.m
% function [kmin,kmax,vmin,vmax] = curv(flag,x,y,z,nrm)
%
% By Dr. Flynn
% Modified by Dirk Colbry 10-20-03
% Modified by Guangpeng Zhang 10-27-07
%
% This function calculates the max and min curvatures of a surface with normal nrm.

[nr nc] = size(nflag);
kmin=zeros(nr,nc);
kmax=zeros(nr,nc);
vmin=zeros(nr,nc,3);
vmax=zeros(nr,nc,3);
A = zeros(nr,nc);
B = zeros(nr,nc);
C = zeros(nr,nc);
cflag = zeros(nr, nc);

% Make this an input parameter
rsz = nb; %5
csz = nb; %5  % half neighborhood size (3 => 7x7 neighborhood)

for i=1:nr
    r0 = max(1,i-rsz);
    r1 = min(nr,i+rsz);
    tnr = r1-r0+1;
%    fprintf(1,'Row %03d of %03d\r',i,nr);
    for j=1:nc
        if nflag(i,j)
            c0=max(1,j-csz);
            c1=min(nc,j+csz);
            tnc=c1-c0+1;
            tnrnc=tnr*tnc;
            tflag=nflag(r0:r1,c0:c1);
            if (sum(sum(tflag)) == tnr*tnc)
                % subtract "current point" to simplyfy calc. of derivatives;
                % won't affect curv estimate
                tx=x(r0:r1,c0:c1)-x(i,j); tx = tx(:);
                ty=y(r0:r1,c0:c1)-y(i,j); ty = ty(:);
                tz=z(r0:r1,c0:c1)-z(i,j); tz = tz(:);
                tz=tz-mean(tz);
                % set up for fit ax^3 + bx^2y + cxy^2 + dy^3 + ex^2 + fxy +
                % gy^2 + hx + iy + j = z
                design = [tx.^3 tx.^2.*ty tx.*ty.^2 ty.^3 tx.^2 tx.*ty ty.^2 tx ty ones(tnr*tnc,1)];

                %dirk code 12/11/03
                %if (rank(design) < 9) error('Rank deficient'); end
                if (rank(design) < 10) 
                    disp('Rank deficient'); 
                else
                    rhs = tz;
                    param = design\rhs;
                    % since we didn't rotate the data, the coordinates of the
                    % principal direction vectors are with respect to the data's x and y axes.
                    a = param(5);
                    b = param(6)/2.0;
                    c = param(7);
                    %Guangpeng code 10-29-2007
                    A(i,j) = 2*a;
                    B(i,j) = 2*b;
                    C(i,j) = 2*c;
                    %dis = sqrt((a-c)^2+2*b^2);
                    % Guangpeng code 10-27-07
                    dis = sqrt((a-c)^2+4*b^2);
                    k1tmp = a+c+dis;
                    k2tmp = a+c-dis;
                    if (abs(k1tmp)>abs(k2tmp))
                        kmin(i,j)=k2tmp;
                        kmax(i,j)=k1tmp;
                    else
                        kmin(i,j)=k1tmp;
                        kmax(i,j)=k2tmp;
                    end
                    % calc directions using Sander & Zucker's quadratic form
                    if (a>=c)
                        e1 = [a-c+dis 2*b];
                        e2 = [-2*b a-c+dis];
                    else
                        e1 = [2*b -(a-c-dis)];
                        e2 = [1-c-dis 2*b];
                    end
                    e1=e1/norm(e1);
                    e2=e2/norm(e2);
                    % e1 and e2 are directions specified in "tangent plane
                    % coordinates".  The basis vectors for these coordinates are
                    % the normalized projections of the X and Y axes on the tangent plane or,
                    % equivalently, the X and Y elementary vectors rotated to span
                    % the tangent plane.
                    nxyz = nrm(i,j,:);
                    nx=nxyz(1);
                    ny=nxyz(2);
                    nz=nxyz(3);
                    bx = [sqrt(1.0-nx^2) 0 -nx];
                    by = [0 sqrt(1.0-ny^2) -ny];    % wow, are these right?
                    vmn=e1 * [bx; by];
                    vmx=e2 * [bx; by];
                    vmin(i,j,:) = vmn/norm(vmn);
                    vmax(i,j,:) = vmx/norm(vmx);
                    cflag(i,j) = 1;
                end
            end
        end
    end
end