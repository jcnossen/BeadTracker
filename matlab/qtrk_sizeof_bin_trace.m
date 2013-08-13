function [nframes, nbeads] = qtrk_sizeof_bin_trace(filename)
    fid = fopen(filename,'r', 'ieee-le'); % little endian byte order
    nbeads = int32(fread(fid, 1, 'uint32'));
        
    % figure out the size of this file and compute number of frames from it
    fseek(fid, 0, 'eof');
    file_size = ftell(fid);
    bytesPerFrame = 4 + 8 + nbeads * 3 * 4;
    nframes = int32((file_size - 4) / bytesPerFrame);
    fclose(fid);
end