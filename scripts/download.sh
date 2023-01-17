#!/bin/bash
out_dir=downloads
mkdir -p ${out_dir}
vid_file=$1
cat ${vid_file} | while read line; do
    vid=${line}
    url=https://www.youtube.com/watch\?v=${vid}
    out_path=${out_dir}/${vid}
    if [[ -f ${out_path}.mp4 ]] || [[ -f ${out_path}.mkv ]] || [[ -f ${out_path}.dummy ]]; then
        echo skip ${out_path}...
        continue
    fi
    echo downloading from $url 
    youtube-dl --rm-cache-dir ${url} -o ${out_path}
    if [[ ! -f ${out_path}.mp4 ]] && [[ ! -f ${out_path}.mkv ]]; then
        echo fail to download ${url}
        touch ${out_path}.dummy
    fi

    sleep 0.5
done

date
echo download finish
