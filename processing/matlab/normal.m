function [nrm, nflag] = normal(flag,x,y,z,nb)
%
% normal.m
% function nrm = normal(flag,x,y,z)
%
% By Dr. Flynn
% Modified by Dirk Colbry 10-20-03
% Calculates the normal of a range surface
% by fitting a plane to the neighboring surface

[nr nc]=size(flag);
nflag = zeros(nr,nc);

nrm=zeros(nr,nc,3);    
rsz = nb;   % row, column size for local neighborhood est
csz = nb;   % rsz=csz=5 --> 11x11 neighborhood
    
for i=1:nr
    r0 = max(1,i-rsz);
    r1 = min(nr,i+rsz);
    tnr = r1-r0+1;
%    fprintf(1,'Doing row %03d of %03d\r',i,nr);
    for j=1:nc
        if flag(i,j)
            c0=max(1,j-csz);
            c1=min(nc,j+csz);
            %fprintf(1,'%d %d; [%d %d]x[%d %d]\n',i,j,r0,r1,c0,c1);
            tnc=c1-c0+1;
            tnrnc=tnr*tnc;
            %tnrnc
            tflag=flag(r0:r1,c0:c1);
            tx=x(r0:r1,c0:c1);
            ty=y(r0:r1,c0:c1);
            tz=z(r0:r1,c0:c1);
            %tx
            %ty
            %tz
            flag2=(tflag>0);
            %flag2
            xx=tx(flag2);
            yy=ty(flag2);
            zz=tz(flag2);
            %xx
            %yy
            %zz
            design = [xx yy zz];  % n x 3
            %size(design)
            rhs = ones(sum(sum(flag2)),1);
            %size(rhs)

            %Dirk Colde 12/11/03
            if (rank(design) < 3) 
                disp('Rank deficient');
            else                
                tn=design\rhs;
                %tn
                tn=tn/norm(tn);
                if (tn(3)<0) tn=-tn; end
                %tn
                nrm(i,j,:) = tn(:);
                nflag(i,j) = 1;
            end
        end
    end
end
%fprintf(1,'...Done.\n');
