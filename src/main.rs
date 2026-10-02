use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling::{context::Context as SwsContext, flag::Flags as SwsFlags};
use image::{RgbImage, ImageBuffer};
use kreuzberg_paddle_ocr::OcrLite;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

// ==================== CLI 参数 ====================

#[derive(Parser, Debug)]
#[command(author, version, about = "从视频中 OCR 提取固定位置字幕并输出 SRT")]
struct Args {
    /// 输入视频文件路径
    #[arg(short, long)]
    input: PathBuf,

    /// 输出 SRT 文件路径
    #[arg(short, long, default_value = "output.srt")]
    output: PathBuf,

    /// 字幕区域起始 Y 坐标占视频高度的比例（0.0 ~ 1.0）
    #[arg(long, default_value = "0.80")]
    roi_y: f32,

    /// 字幕区域高度占视频高度的比例
    #[arg(long, default_value = "0.20")]
    roi_h: f32,

    /// 每秒采样帧数
    #[arg(long, default_value = "3")]
    fps_sample: u32,

    /// ONNX Runtime 线程数
    #[arg(long, default_value = "2")]
    ocr_threads: u32,

    /// 模型目录路径（kreuzberg-paddle-ocr 会自动下载到该目录）
    #[arg(long, default_value = "./models")]
    model_dir: PathBuf,
}

// ==================== OCR 结果 ====================

#[derive(Debug, Clone)]
struct FrameOcrResult {
    frame_index: usize,
    time_secs: f64,
    text: String,
    confidence: f32,
}

// ==================== 字幕条目 ====================

#[derive(Debug, Clone)]
struct SubtitleEntry {
    start_secs: f64,
    end_secs: f64,
    text: String,
}

impl SubtitleEntry {
    fn format_timestamp(secs: f64) -> String {
        let total_ms = (secs * 1000.0).round() as u64;
        let hours = total_ms / 3_600_000;
        let minutes = (total_ms % 3_600_000) / 60_000;
        let seconds = (total_ms % 60_000) / 1000;
        let millis = total_ms % 1000;
        format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
    }
}

// ==================== 视频解码 ====================

fn decode_video_frames(
    input: &PathBuf,
    roi_y_ratio: f32,
    roi_h_ratio: f32,
    fps_sample: u32,
) -> Result<Vec<(usize, f64, RgbImage)>> {
    ffmpeg::init().context("初始化 FFmpeg 失败")?;

    let mut ictx = ffmpeg::format::input(input).context("打开视频文件失败")?;
    let input_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .context("未找到视频流")?;
    let video_stream_index = input_stream.index();

    let time_base = input_stream.time_base();
    let frame_rate = input_stream.rate();

    let decoder_ctx = ffmpeg::codec::context::Context::from_parameters(input_stream.parameters())?;
    let mut decoder = decoder_ctx.decoder().video()?;

    let src_width = decoder.width();
    let src_height = decoder.height();

    let roi_y = (src_height as f32 * roi_y_ratio) as u32;
    let roi_h = (src_height as f32 * roi_h_ratio) as u32;
    let roi_h = roi_h.min(src_height - roi_y);

    let sample_interval = ((frame_rate.denominator() as f64 / frame_rate.numerator() as f64)
        * fps_sample as f64)
        .round()
        .max(1.0) as usize;

    let mut scaler = SwsContext::get(
        decoder.format(),
        src_width,
        src_height,
        Pixel::RGB24,
        src_width,
        src_height,
        SwsFlags::BILINEAR,
    )?;

    let mut frames: Vec<(usize, f64, RgbImage)> = Vec::new();
    let mut frame_index: usize = 0;
    let mut rgb_frame = ffmpeg::util::frame::Video::empty();

    let mut packet_iter = ictx.packets();
    while let Some((stream, packet)) = packet_iter.next() {
        if stream.index() != video_stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;

        loop {
            match decoder.receive_frame(&mut rgb_frame) {
                Ok(()) => {
                    if frame_index % sample_interval == 0 {
                        let mut rgb = ffmpeg::util::frame::Video::empty();
                        scaler.run(&rgb_frame, &mut rgb)?;

                        let data = rgb.data(0);
                        let stride = rgb.stride(0);

                        let mut img: RgbImage = ImageBuffer::new(src_width, src_height);
                        for y in 0..src_height as usize {
                            let row_start = y * stride;
                            let src_row = &data[row_start..row_start + (src_width as usize * 3)];
                            
                            let dst_slice = img.as_flat_samples_mut().samples;
                            let dst_row_start = y * (src_width as usize * 3);
                            let dst_row_end = dst_row_start + (src_width as usize * 3);
                            dst_slice[dst_row_start..dst_row_end].copy_from_slice(src_row);
                            // let dst_row = &mut img.as_mut().as_mut_slice()
                            //     [y * (src_width as usize * 3)..(y + 1) * (src_width as usize * 3)];
                            // dst_row.copy_from_slice(src_row);
                        }

                        let roi_img = image::imageops::crop_imm(
                            &img,
                            0,
                            roi_y,
                            src_width,
                            roi_h,
                        )
                        .to_image();

                        let pts = rgb_frame.pts().unwrap_or(0);
                        let time_secs =
                            pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;

                        frames.push((frame_index, time_secs, roi_img));
                    }
                    frame_index += 1;
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(ffmpeg::Error::Other { errno: ffmpeg::error::EAGAIN }) => break,
                Err(e) => return Err(e.into()),
            }
        }
    }

    decoder.send_eof()?;
    loop {
        match decoder.receive_frame(&mut rgb_frame) {
            Ok(()) => {
                if frame_index % sample_interval == 0 {
                    let mut rgb = ffmpeg::util::frame::Video::empty();
                    scaler.run(&rgb_frame, &mut rgb)?;
                    let data = rgb.data(0);
                    let stride = rgb.stride(0);
                    let mut img: RgbImage = ImageBuffer::new(src_width, src_height);
                    for y in 0..src_height as usize {
                        let row_start = y * stride;
                        let src_row = &data[row_start..row_start + (src_width as usize * 3)];
                        
                        let dst_slice = img.as_flat_samples_mut().samples;
                            let dst_row_start = y * (src_width as usize * 3);
                            let dst_row_end = dst_row_start + (src_width as usize * 3);
                            dst_slice[dst_row_start..dst_row_end].copy_from_slice(src_row);
                            // let dst_row = &mut img.as_mut().as_mut_slice()
                            //     [y * (src_width as usize * 3)..(y + 1) * (src_width as usize * 3)];
                            // dst_row.copy_from_slice(src_row);
                    }
                    let roi_img =
                        image::imageops::crop_imm(&img, 0, roi_y, src_width, roi_h).to_image();
                    let pts = rgb_frame.pts().unwrap_or(0);
                    let time_secs =
                        pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;
                    frames.push((frame_index, time_secs, roi_img));
                }
                frame_index += 1;
            }
            Err(_) => break,
        }
    }

    Ok(frames)
}

// ==================== 并行 OCR ====================

fn ocr_frames_parallel(
    frames: Vec<(usize, f64, RgbImage)>,
    model_dir: &PathBuf,
    ocr_threads: u32,
) -> Result<Vec<FrameOcrResult>> {
    // kreuzberg-paddle-ocr 会自动下载模型到指定目录
    // 首次调用 init_models 时，模型会自动下载并缓存
    let mut ocr = OcrLite::new();
    ocr.init_models(
        model_dir.to_str().unwrap(),
        model_dir.to_str().unwrap(),
        model_dir.to_str().unwrap(),
        ocr_threads as usize,
    )
    .context("初始化 OCR 模型失败")?;

    // OcrLite 的 &self 方法支持无锁并发，可以安全地共享实例
    let ocr = Arc::new(ocr);

    let results: Vec<FrameOcrResult> = frames
        .into_par_iter()
        .map(|(frame_index, time_secs, roi_img)| {
            let ocr = Arc::clone(&ocr);

            // 执行 OCR（&self 方法，线程安全）
            let result = ocr
                .detect(
                    &roi_img,
                    50,    // padding
                    1024,  // max_side_len
                    0.5,   // box_score_thresh
                    0.3,   // box_thresh
                    1.6,   // unclip_ratio
                    true,  // do_angle
                    false, // most_angle
                )
                .expect("OCR 检测失败");

            let mut text = String::new();
            let mut total_conf = 0.0f32;
            let mut count = 0usize;
            for block in &result.text_blocks {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&block.text);
                total_conf += block.text_score;
                count += 1;
            }
            let avg_conf = if count > 0 { total_conf / count as f32 } else { 0.0 };

            FrameOcrResult {
                frame_index,
                time_secs,
                text: text.trim().to_string(),
                confidence: avg_conf,
            }
        })
        .collect();

    let mut results = results;
    results.sort_by_key(|r| r.frame_index);
    Ok(results)
}

// ==================== 时序合并为字幕段 ====================

fn merge_to_subtitles(
    ocr_results: &[FrameOcrResult],
    min_confidence: f32,
    video_duration: f64,
) -> Vec<SubtitleEntry> {
    let mut subtitles: Vec<SubtitleEntry> = Vec::new();
    let mut current_text = String::new();
    let mut current_start: f64 = 0.0;
    let mut last_time: f64 = 0.0;

    for r in ocr_results {
        let text = if r.confidence < min_confidence {
            String::new()
        } else {
            r.text.clone()
        };

        if text != current_text {
            if !current_text.is_empty() {
                subtitles.push(SubtitleEntry {
                    start_secs: current_start,
                    end_secs: r.time_secs,
                    text: current_text.clone(),
                });
            }
            if !text.is_empty() {
                current_start = r.time_secs;
            }
            current_text = text;
        }
        last_time = r.time_secs;
    }

    if !current_text.is_empty() {
        subtitles.push(SubtitleEntry {
            start_secs: current_start,
            end_secs: video_duration.max(last_time),
            text: current_text,
        });
    }

    subtitles
}

// ==================== 输出 SRT ====================

fn write_srt(subtitles: &[SubtitleEntry], output: &PathBuf) -> Result<()> {
    let mut srt = String::new();
    for (i, sub) in subtitles.iter().enumerate() {
        srt.push_str(&format!("{}\n", i + 1));
        srt.push_str(&format!(
            "{} --> {}\n",
            SubtitleEntry::format_timestamp(sub.start_secs),
            SubtitleEntry::format_timestamp(sub.end_secs)
        ));
        srt.push_str(&format!("{}\n\n", sub.text));
    }
    std::fs::write(output, srt).context("写入 SRT 文件失败")?;
    Ok(())
}

// ==================== 主函数 ====================

fn main() -> Result<()> {
    let args = Args::parse();

    println!("正在解码视频: {:?}", args.input);
    let frames = decode_video_frames(&args.input, args.roi_y, args.roi_h, args.fps_sample)?;
    println!("已提取 {} 个采样帧", frames.len());

    if frames.is_empty() {
        println!("未提取到任何帧，退出。");
        return Ok(());
    }

    let video_duration = frames.last().map(|(_, t, _)| *t).unwrap_or(0.0);

    println!("开始并行 OCR 识别...");
    let ocr_results = ocr_frames_parallel(frames, &args.model_dir, args.ocr_threads)?;
    println!("OCR 完成，共识别 {} 帧", ocr_results.len());

    let subtitles = merge_to_subtitles(&ocr_results, 0.5, video_duration);
    println!("生成 {} 条字幕", subtitles.len());

    write_srt(&subtitles, &args.output)?;
    println!("已保存到: {:?}", args.output);

    Ok(())
}