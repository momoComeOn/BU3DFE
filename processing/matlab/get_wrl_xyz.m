function data = get_wrl_xyz(path)

pc = fopen(path, 'r');

for i = 1:20
    line = fgetl(pc);
end

x = [];
y = [];
z = [];

while (line(end) ~= ']')
    if line(end) == ','
        line = deblank(line(1:end-1));
    else
        line = deblank(line);
    end
    S = regexp(line, ' ', 'split');
    X = str2double(char(S(end-2)));
    Y = str2double(char(S(end-1)));
    Z = str2double(char(S(end)));
    x  =[x; X];
    y = [y; Y];
    z = [z; Z];
    line = fgetl(pc);
end
data = [x, y, z];
fclose(pc);
