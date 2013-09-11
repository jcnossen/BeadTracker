function d = normalize(d)

d = double(d);

minV = min(d(:));
maxV = max(d(:));

d = (d-minV)/(maxV-minV);

end

