function mat = lvreadbinarray(filename, dim, elemtype)

    if nargin<2
        dim=2; % assume 2D array
    end

    if nargin<3
        elemtype = 'single';
    end

    fid = fopen(filename,'r', 'ieee-le'); % little endian

    if fid ~= -1
        s = fread(fid, dim, 'int32');
        mat = fread(fid, [s(2) s(1)], elemtype);
        %d = zeros(s(2),s(1));
        %for k=1:s(2)
        
        fclose(fid);
    end


end