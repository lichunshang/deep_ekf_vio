function ts = loadTimestamps(ts_dir,numeric)

ts = [];

fid = fopen([ts_dir '/timestamps.txt']);

if fid~=-1
  
  col = textscan(fid,'%s\n',-1,'delimiter',',');
  ts = col{1};
  fclose(fid);

  if nargin==2
    for i=1:length(ts)
      num(i) = stringToTimestampMex(ts{i});
    end
    ts = num;
  end
end
