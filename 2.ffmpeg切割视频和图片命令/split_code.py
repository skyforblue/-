# 切割图片
ffmpeg -i 20190924070000.mp4 -ss 50 -f test.jpg

# 切割视频
ffmpeg -ss 00:54:00 -t 350 -i Bar03_20190817130000.mp4  -c:v libx264 -c:a aac -strict experimental -b:a 98k Bar03_20190817130000_2.mp4

# 拼接视频
# mylist.txt
file '/path/to/file1'
file '/path/to/file2'
file '/path/to/file3'
# 拼接命令
ffmpeg -f concat -i mylist.txt -c copy output
# 直接命令拼接，有时会失效
ffmpeg -i "concat:input1.mpg|input2.mpg|input3.mpg" -c copy output.mpg
