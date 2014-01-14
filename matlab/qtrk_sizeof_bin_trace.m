function [nframes, nbeads, ninfocol, colnames, data_offset] = qtrk_sizeof_bin_trace(filename)
    fid = fopen(filename,'r', 'ieee-le'); % little endian byte order
    if fid<0,
        fprintf('can ''t open file %s', filename);
        nframes=0;
        return
    end
    
    version = int32(fread(fid, 1, 'int32'));
    expected_version = 2;
    
    if version ~= expected_version
        fprintf('File has unknown version (%d), expecting version %d\n', version, expected_version);
    end
    
    nbeads = int32(fread(fid, 1, 'int32'));
    ninfocol = int32(fread(fid, 1, 'int32'));
    data_offset = int32(fread(fid, 1, 'int32'));
    
    for k=1:ninfocol
        colnames{k} = read_zero_term_string(fid);
    end

    % figure out the size of this file and compute number of frames from it
    fseek(fid, 0, 'eof');
    file_size = ftell(fid);
    bytesPerFrame = 4 + 8 + ninfocol * 4 + nbeads * 4 * 4;
    nframes = int32((file_size - data_offset) / bytesPerFrame);
    fclose(fid);
end

function s = read_zero_term_string(fid)
    s= char;
    while ~feof(fid)
        ch = fread(fid, 1, 'int8');
        if ch == 0, break, end
        s = [s ch];
    end
end
