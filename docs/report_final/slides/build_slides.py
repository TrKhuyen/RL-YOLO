"""Build editable PPTX and matching HTML using Python standard library + Pillow.
Run: python3 docs/report_final/slides/build_slides.py
"""
from pathlib import Path
import re, html, zipfile, textwrap
from PIL import ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
W,H=1600,900
BG='F6F8F7'; INK='17332F'; MUTED='5E726E'; GREEN='17836B'; BLUE='286BC0'; PURPLE='7954B4'; ORANGE='B87324'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
slides=[]

def esc(s): return html.escape(str(s),quote=True)
def wrap(s,size,width,bold=False):
 f=ImageFont.truetype(BOLD if bold else FONT,size)
 out=[]
 for para in str(s).split('\n'):
  line=''
  for word in para.split():
   trial=(line+' '+word).strip()
   if f.getlength(trial)>width and line: out.append(line);line=word
   else:line=trial
  out.append(line)
 return out
class Slide:
 def __init__(self,title,section='DỮ LIỆU',color=GREEN,tag=None):
  self.items=[]; self.title=title; self.section=section;self.color=color;self.tag=tag;self.notes='';slides.append(self)
  self.rect(0,0,W,H,BG)
  self.rect(0,0,12,H,color)
  self.text(section,64,35,1400,30,18,color,True)
  self.text(title,64,87,1472,104,38,INK,True)
  self.rect(64,192,1472,2,'DCE5E1')
  self.text('RL–YOLO  /  BÁO CÁO TIẾN ĐỘ',64,860,1200,22,13,MUTED)
  self.text(f'{len(slides):02d}',1460,850,76,32,20,color,True)
 def rect(self,x,y,w,h,fill,line=None,r=0): self.items.append(dict(t='rect',x=x,y=y,w=w,h=h,fill=fill,line=line,r=r))
 def text(self,text,x,y,w,h,size=25,color=INK,bold=False,align='left'):
  lines=wrap(text,size,w-4,bold)
  if len(lines)*size*1.33>h+3: raise ValueError(f'Overflow slide {len(slides)}: {text[:50]} {len(lines)*size*1.33}>{h}')
  self.items.append(dict(t='text',x=x,y=y,w=w,h=h,text='\n'.join(lines),size=size,color=color,bold=bold,align=align))
 def takeaway(self,text):
  self.rect(64,772,1472,64,self.color,r=12)
  self.text(text,84,784,1430,43,23,'FFFFFF',True)
 def card(self,x,y,w,h,label,body,size=25):
  self.rect(x,y,w,h,'FFFFFF','DCE5E1',14)
  self.rect(x,y,5,h,self.color)
  self.text(label,x+24,y+21,w-48,75,27,self.color,True)
  self.text(body,x+24,y+103,w-48,h-118,size)
 def bullets(self,items,x,y,w,size=26,gap=18):
  for s in items:
   lines=wrap(s,size,w-30)
   h=len(lines)*size*1.33
   self.text('•',x,y,25,40,size,self.color,True)
   self.text(s,x+30,y,w-30,h+3,size)
   y+=h+gap
 def table(self,headers,rows,x=64,y=238,width=1472,rowh=64,colwidths=None,size=23):
  weights=colwidths or [1]*len(headers); widths=[width*v/sum(weights) for v in weights]
  for ri,row in enumerate([headers]+rows):
   xx=x
   for ci,(v,cw) in enumerate(zip(row,widths)):
    self.rect(xx,y+ri*rowh,cw-2,rowh-2,self.color if ri==0 else ('FFFFFF' if ri%2 else 'EBF0ED'))
    self.text(v,xx+14,y+ri*rowh+12,cw-28,rowh-17,size,'FFFFFF' if ri==0 else INK,ri==0 or ci==0)
    xx+=cw
 def flow(self,labels,y=285,x=64,width=1472,height=150,size=23):
  n=len(labels); gap=35; cw=(width-gap*(n-1))/n
  for i,l in enumerate(labels):
   xx=x+i*(cw+gap)
   self.rect(xx,y,cw,height,'FFFFFF','D3DED8',12)
   lines=wrap(l,size,cw-24,True)
   self.text(l,xx+12,y+(height-len(lines)*size*1.33)/2,cw-24,height-12,size,self.color,True,'center')
   if i<n-1:self.text('→',xx+cw+4,y+height/2-22,30,55,31,self.color,True)

def new(title,section='DỮ LIỆU',color=GREEN):return Slide(title,section,color)
def two(title,left,right,section='DỮ LIỆU',color=GREEN,take=None):
 s=new(title,section,color);s.card(64,235,712,490,*left);s.card(800,235,736,490,*right)
 if take:s.takeaway(take)
 return s

s=new('Tối ưu mô hình YOLO phát hiện lá sâu bệnh', 'BÁO CÁO TIẾN ĐỘ  /  ĐỀ TÀI',GREEN)
s.rect(64,207,1472,554,INK,r=20)
s.text('BẰNG HỌC TĂNG CƯỜNG\nVÀ XÂY DỰNG WEBSITE NHẬN DIỆN',105,255,1360,154,43,'FFFFFF',True)
s.text('PlantDOC  →  Ba kịch bản tối ưu  →  Ứng dụng website',105,468,1330,86,29,'ABDCCB')
s.text('[Tên nhóm]  ·  [Thành viên]\nGVHD: [Điền tên]    |    Ngày báo cáo: [Điền ngày]',105,618,1330,103,25,'FFFFFF')

s=two('Chuẩn bị dữ liệu: lựa chọn PlantDOC',('Dữ liệu được chọn','PlantDOC — bản đã làm sạch trong dự án.\n\nBài toán detection: tìm vị trí lá và class tương ứng.'),('Biểu diễn nhãn','28 class: loại cây và tình trạng lá khỏe/bệnh.\n\nNhãn YOLO: class, cx, cy, width, height.'),take='Các thống kê mô tả bản dữ liệu của dự án, không phải quy mô PlantDOC nguyên bản.')

s=new('Bộ dữ liệu thực nghiệm: 3.129 ảnh, 28 class')
s.table(['Split','Ảnh','Bounding box','Vai trò'],[['Train','2.411','10.333','Học trọng số; tạo preference'],['Validation','479','1.887','Chọn cấu hình / checkpoint'],['Test','239','1.063','Đánh giá lựa chọn đã khóa'],['Tổng','3.129','13.283','28 class']],rowh=73,colwidths=[1.1,.8,1.2,3],size=24)
s.text('Audit source ID: chưa thấy nguồn chung giữa split theo quy tắc bỏ hậu tố augment.\nGiới hạn: chưa loại trừ mọi near-duplicate khác ID.',64,638,1472,100,23,MUTED)
s.takeaway('Train có 667 ảnh mang hậu tố augmentation đã nhận diện; số ảnh không bằng số nguồn độc lập.')

s=new('Trực quan hóa dữ liệu trước augmentation')
for x,label in [(64,'BIỂU ĐỒ PHÂN BỐ CLASS'),(812,'ẢNH DỮ LIỆU CÓ BOUNDING BOX')]:
 s.rect(x,240,724,370,'FFFFFF','A9C5B8',15);s.text('NHÓM TỰ BỔ SUNG',x+26,340,672,55,29,GREEN,True,'center');s.text(label,x+26,411,672,75,23,MUTED,False,'center')
s.text('Nhận xét của nhóm: [class ít mẫu / ảnh nhiều lá / nền / kích thước đối tượng]',64,650,1472,80,25)
s.takeaway('Biểu đồ cần ghi rõ đếm ảnh hay box và thuộc split nào.')

s=two('Augmentation: mục tiêu và hai hình thức',('Offline — biến thể được lưu','667 ảnh train có hậu tố flip/rotation được nhận diện.\n\n1.744 file train còn lại không mang các hậu tố đã biết.\n\nẢnh và nhãn augment phải cùng split train.'),('Online — khi đọc ảnh','Tạo biến thiên input qua các lượt huấn luyện.\n\nKết hợp hình học và quang học.\n\nRecipe phụ thuộc từng kịch bản.'),take='Tăng biến thiên input; chưa có ablation để quy gain AP cho riêng từng phép augment.')

s=new('Phương pháp augmentation: ảnh đi cùng nhãn')
s.table(['Nhóm','Phép biến đổi','Xác suất / phạm vi'],[['Kích thước','Resize cạnh dài; pad 114','640 × 640; luôn áp dụng'],['Ánh sáng','Brightness / contrast','p=0.5; ±0.3'],['Màu sắc','Hue / saturation / value','p=0.4; ±15 / ±30 / ±20'],['Nhiễu / mờ','Gaussian / motion blur','p=0.3 / 0.2'],['Hình học','Lật ngang / dọc; xoay','p=0.5 / 0.3; xoay p=0.4, ±15°'],['Cục bộ','CLAHE','p=0.2; clip limit 1–2']],rowh=62,colwidths=[1,1.5,2],size=22)
s.text('Recipe online stage hai KB1. BBox biến đổi đồng bộ; clipping; min_visibility=0.3.\nKB2: resize/pad xác định để giữ slot/box preference. Val/test: không random augment.',64,685,1472,78,21,MUTED)
s.takeaway('Một ảnh có thể nhận nhiều phép; normalize [0,1] rồi chuyển tensor.')

s=new('Trực quan hóa ảnh sau augmentation')
labels=['ẢNH GỐC + BOX','HÌNH HỌC + BOX','QUANG HỌC + BOX','KẾT HỢP + BOX']
for i,label in enumerate(labels):
 x=64+i*374;s.rect(x,250,350,365,'FFFFFF','A9C5B8',14);s.text('TỰ THÊM\nẢNH',x+24,350,302,115,31,GREEN,True,'center');s.text(label,x+10,635,330,72,20,INK,True,'center')
s.takeaway('Dùng cùng một ảnh train để so; ghi tên phép và tham số biến đổi thực tế.')

s=new('Ba hướng tối ưu được nghiên cứu độc lập','TỔNG QUAN',GREEN)
for x,label,body,c in [(64,'KB1 · REWARD','Fine-tune Detect head bằng native loss, proxy và task reward.\n\nRL-inspired; có kết quả test.',BLUE),(563,'KB2 · DPO','Học ưu tiên giữ/bỏ candidate từ GT và reference cố định.\n\nHiện có validation.',PURPLE),(1062,'KB3 · HP','A: cấu hình bằng TPE.\nB: controller PPO trong train.\n\nA là HPO; B là RL.',ORANGE)]:
 old=s.color;s.color=c;s.card(x,235,474,475,label,body,24);s.color=old
s.takeaway('KB1 → KB2 → KB3 là thứ tự trình bày; không phải chuỗi fine-tune nối tiếp.')

s=two('KB1 — Mục tiêu và câu hỏi nghiên cứu',('Mục tiêu','Kiểm chứng quy trình hai giai đoạn để cải thiện detector so với checkpoint supervised tương ứng.\n\nMetric chính: canonical mAP50–95.'),('Câu hỏi nghiên cứu','RQ1: thêm reward-guided cải thiện bao nhiêu so S0?\n\nRQ2: kiến trúc guided nào tốt nhất theo validation?'),'KB1  /  MỤC TIÊU',BLUE,'Câu hỏi bổ sung: gain đến từ continuation, proxy hay reward?')

s=new('KB1 — Ý tưởng và objective','KB1  /  PHƯƠNG PHÁP',BLUE)
s.flow(['Prediction train','Matching với GT','Reward P / R / IoU','Advantage → loss'],y=235,height=125)
s.text('Reward tại IoU 0.5 / 0.6 / 0.7 / 0.8; trọng số P / R / IoU = 0.25 / 0.45 / 0.30.',64,397,1472,77,26)
s.rect(64,505,1472,135,'FFFFFF','DCE5E1',12)
s.text('L = 0.25 Lnative + 0.10 Lreward + 0.25 Lproxy + 0.001 LSP',90,542,1420,65,31,BLUE,True,'center')
s.text('Proxy: confidence TP/FP/FN.  LSP: hạn chế drift khỏi S0.  Advantage: EMA reward baseline.',64,677,1472,65,23,MUTED)
s.takeaway('Reward-guided / RL-inspired; implementation chưa phải REINFORCE chuẩn.')

s=new('KB1 — Kiến trúc tổng thể','KB1  /  KIẾN TRÚC',BLUE)
s.flow(['Pretrained','Supervised','S0 của model','Fine-tune head','Best validation','Test đã khóa'],y=245,height=135,size=22)
s.text('Backbone / neck và BN statistics đóng băng trong stage hai',64,416,1472,50,28,BLUE,True,'center')
for x,label,body in [(64,'S0','Không train thêm'),(563,'S1','Native + L2-SP'),(1062,'G','Native + proxy + reward + L2-SP')]:
 s.rect(x,516,474,164,'FFFFFF','DCE5E1',12);s.text(label,x+20,532,434,45,29,BLUE,True);s.text(body,x+20,590,434,77,25)
s.takeaway('Bảy model; mỗi model dùng S0 riêng, các nhánh có cùng điểm khởi đầu.')

s=two('KB1 — Thiết kế thực nghiệm đã hoàn tất',('Cohort chính','7 model: YOLOv5s, YOLOv8n/s, YOLOv11n/s, YOLO26n, DP-YOLO.\n\nContinuation seed 42/43/44.\n\nBảng chính dùng seed Ubuntu 43/44.'),('Ablation kiểm soát','YOLOv11n và YOLOv8s; 3 seed.\n\nS1, proxy-only, reward-only, G, G bỏ L2-SP.\n\n10.000 micro-step = 5.000 optimizer update; so last.'),'KB1  /  THỰC NGHIỆM',BLUE,'Đã hoàn tất 58 lượt train dài bổ sung; continuation và ablation tổng hợp riêng.')

s=new('KB1 — Guided cải thiện so S0 ở cả bảy model','KB1  /  KẾT QUẢ TEST',BLUE)
s.table(['Model','S0 mAP (%)','G mAP (%)','Δ (đpt)'],[['YOLOv5s','30.398','31.261','+0.863'],['YOLOv8n','33.201','33.561','+0.359'],['YOLOv8s','33.976','35.387','+1.412'],['YOLOv11n','37.534','37.768','+0.234'],['YOLOv11s','36.271','36.442','+0.172'],['YOLO26n','33.357','33.831','+0.474'],['DP-YOLO','28.913','29.366','+0.453']],width=1030,rowh=61,colwidths=[1.3,1,1,.85],size=23)
s.text('YOLOv11n\nGuided dẫn đầu\nVal: 40.648%\nTest: 37.768%\n\nYOLOv8s\nGain lớn nhất\n+1.412 đpt',1140,253,395,450,29,BLUE,True)
s.takeaway('Mean test seed 43/44. G > S0 ở 21/21 cặp model–seed quan sát, gồm 42/43/44.')

s=two('KB1 — Đóng góp riêng của reward',('Đọc đúng các phép so','G − S0: toàn bộ stage hai.\n\nG − S1: proxy + reward.\n\nG − proxy-only: phần thêm của reward khi đã có proxy.'),('Ablation cùng 10.000 micro-step','G − proxy-only, mean test 3 seed:\n\nYOLOv11n: +0.070727 đpt\n\nYOLOv8s: −0.049986 đpt\n\nG hơn S1 ở 16/21 cặp continuation.'),'KB1  /  PHÂN TÍCH',BLUE,'Lợi ích riêng của reward phụ thuộc kiến trúc và protocol; không quy toàn bộ gain cho reward.')

s=new('KB1 — Định hướng tương lai','KB1  /  TIẾP THEO',BLUE)
for y,n,title,body in [(235,'01','Tăng bằng chứng','Nhiều S0 supervised độc lập; nguồn ảnh hoặc tập đánh giá khác.'),(407,'02','Đo độ bất định','Phân tích lỗi theo ảnh/source; paired bootstrap AP phù hợp.'),(579,'03','Cải thiện objective','Chẩn đoán gradient/advantage; FN surrogate gắn với đúng GT bị bỏ sót.')]:
 s.rect(64,y,1472,147,'FFFFFF','DCE5E1',12);s.text(n,90,y+33,90,70,37,BLUE,True);s.text(title,210,y+20,1290,45,29,BLUE,True);s.text(body,210,y+78,1270,60,25)
s.takeaway('KB1 đã hoàn tất phạm vi hiện tại; các mục này là nghiên cứu mở rộng.')

s=two('KB2 — Mục tiêu và câu hỏi nghiên cứu',('Mục tiêu','Học quyết định giữ/bỏ candidate bằng preference tự động do GT chấm.\n\nModel xác nhận: YOLOv8n.\n\nKhởi tạo: S0 supervised KB1.'),('Câu hỏi nghiên cứu','RQ1: Native + DPO cải thiện gì so S0?\n\nRQ2: DPO có hơn native-only cùng ngân sách?\n\nRQ3: đánh đổi AP, recall, FP/FN và class?'),'KB2  /  MỤC TIÊU',PURPLE,'Preference có giám sát bởi GT; không phải nhãn sở thích của con người.')

s=new('KB2 — Hai phương án chỉ khác một quyết định','KB2  /  Ý TƯỞNG',PURPLE)
s.flow(['Ảnh + context S0','A: giữ candidate j','NMS toàn ảnh','GT → chosen / rejected'],y=239,height=125,size=23)
s.flow(['Cùng box/class/score','B: bỏ candidate j','NMS toàn ảnh','So hậu quả với A'],y=410,height=125,size=23)
s.bullets(['Keep phải cứu/giữ GT mục tiêu; drop phải thực sự giảm FP.','Gate toàn ảnh bảo vệ GT đang phát hiện và không cho tăng FP.'],64,588,1472,27,15)
s.takeaway('Học quyết định cục bộ, kiểm hậu quả toàn ảnh; inference thực tế không cần GT.')

s=new('KB2 — Kiến trúc policy và reference','KB2  /  KIẾN TRÚC',PURPLE)
s.flow(['S0 supervised','Policy: học cv3','Native loss\nbackground 0.5','AdamW → cv3'],y=235,height=125,size=24)
s.flow(['GT → A/B → bank','2 preference +\n2 uniform ảnh','DPO logit\npolicy / reference','Lnative + 0.1 LDPO'],y=416,height=125,size=23)
s.rect(64,595,1472,128,'FFFFFF','DCE5E1',12)
s.text('Reference = S0 cố định. Backbone/neck, regression và BN statistics đóng băng.\nValidation dùng ordinary YOLO inference; hiện chưa trực tiếp học sửa tọa độ box.',89,620,1422,92,25)
s.takeaway('Native học GT trên 4 ảnh; DPO bổ sung 2 quyết định/update.')

s=two('KB2 — Policy và thiết kế đối chứng',('Policy Bernoulli','q(keep) = sigmoid(z)\nq(drop) = 1 − sigmoid(z)\n\nm = s(zpolicy − zreference)\nLDPO = softplus(−βm)\n\ns = +1 keep; −1 drop. β=0.1; λ=0.1.'),('Đối chứng ghép seed','S0: không train thêm.\nNative-only: λDPO=0; background 0.5.\nNative+DPO: λDPO=0.1.\n\n3 seed × 2 nhánh × 15.000 update.\nCùng stream ảnh/pair trong mỗi seed.\nSo best và so cùng last 15k.'),'KB2  /  THỰC NGHIỆM',PURPLE,'Probability chuẩn hóa cho keep/drop, chưa phải likelihood toàn detection set sau NMS.')

s=new('KB2 — AP tăng lặp lại trên validation','KB2  /  KẾT QUẢ',PURPLE)
s.text('YOLOv8n  ·  Validation  ·  Seed 42/43/44  ·  Mean ± SD',64,213,1472,40,23,MUTED)
s.table(['So sánh','Native (%)','DPO (%)','Δ (đpt)'],[['Best từng run','38.1196 ± 0.0608','38.1707 ± 0.0429','+0.0511 ± 0.0202'],['Cùng last 15k','38.0595 ± 0.0430','38.1670 ± 0.0475','+0.1075 ± 0.0337']],y=283,rowh=88,colwidths=[1.1,1.4,1.4,1.4],size=24)
s.text('S0: 37.8218%    →    Best DPO: +0.3489 đpt so S0',64,594,1472,62,32,PURPLE,True)
s.text('DPO hơn native về mAP / AP50 / AP75 ở cả ba seed; mức tăng thêm còn nhỏ.',64,685,1472,60,25)
s.takeaway('Đây là kết quả validation; chuỗi thí nghiệm KB2 hiện chưa đánh giá hiệu năng test.')

s=new('KB2 — AP tăng, recall tại ngưỡng cố định giảm','KB2  /  ĐÁNH ĐỔI',PURPLE)
s.table(['Metric','S0','DPO mean','Thay đổi'],[['Precision (%)','50.9227','53.1581','+2.2354 đpt'],['Recall (%)','54.1070','53.0825','−1.0246 đpt'],['F1 (%)','52.4666','53.1200','+0.6534 đpt'],['FP','984','882.67','−101.33'],['FN','866','885.33','+19.33']],width=1060,rowh=73,colwidths=[1.2,1,1,1.2],size=24)
s.text('Confidence 0.25\nMatching IoU 0.5\n\nMean class AP:\n12 tăng / 16 giảm\n\nAP small:\n4.3965 → 4.0525%',1170,252,365,465,26,PURPLE,True)
s.takeaway('Ranking tốt hơn chưa đồng nghĩa ít bỏ sót hơn ở mọi threshold; geometry vẫn đóng băng.')

s=new('KB2 — Ưu tiên kiểm chứng trước khi mở rộng','KB2  /  TIẾP THEO',PURPLE)
for x,label,body in [(64,'01 · XÁC NHẬN','Khóa recipe/selector.\n\nHoldout/test cùng S0 và native.\n\nThêm S0 độc lập khi cần.'),(563,'02 · ĐỐI CHỨNG','DPO so BCE có/không reference.\n\nGradient-strength matching.\n\nSweep β / λ / background.'),(1062,'03 · MỞ RỘNG','Bank/class coverage.\n\nAdapter model khác.\n\nGeometry policy; augment ánh xạ đúng candidate.')]:s.card(x,235,474,490,label,body,25)
s.takeaway('Mục tiêu: gain AP bền vững hơn, giảm đánh đổi recall và cải thiện độ phủ.')

s=two('KB3 — Mục tiêu và câu hỏi nghiên cứu',('Mục tiêu','Tìm cấu hình hoặc chính sách điều chỉnh hyperparameter để cải thiện chất lượng detector.\n\nĐối chứng cùng init, recipe, evaluator, seed và ngân sách.'),('Hai câu hỏi','RQ-A: local HPO có hơn anchor và supervised KB1?\n\nRQ-B: controller RL có hơn cấu hình/lịch cố định?'),'KB3  /  MỤC TIÊU',ORANGE,'A tạo mốc HPO; B nghiên cứu điều khiển thích nghi bằng RL.')

s=two('KB3 — Chọn trước run hay điều chỉnh trong run?',('A — TPE / HPO','History → đề xuất LR/WD → train detector mới → validation → cập nhật history.\n\nBase parameters chọn trước trial; warmup/decay theo lịch đã định.'),('B — PPO / RL','Quan sát training → action đổi HP → train segment tiếp → reward → cập nhật controller.\n\nMột episode là một trainer liên tục.'),'KB3  /  Ý TƯỞNG',ORANGE,'TPE là HPO truyền thống; PPO là hướng RL cần kiểm chứng chất lượng policy.')

s=new('KB3 — Tách recipe, pretrained và baseline','KB3  /  KIẾN TRÚC',ORANGE)
s.flow(['YAML KB1\nRecipe','COCO pretrained gốc\nKhởi tạo trial','Best supervised KB1\nĐối chứng chỉ đọc'],y=235,height=130,size=24)
s.flow(['TPE → cấu hình','Detector train mới','Native chọn checkpoint','Canonical chấm trial'],y=425,height=130,size=23)
s.text('B (nghiên cứu tiếp): controller ↔ live trainer ↔ metrics/reward',64,623,1472,70,29,ORANGE,True,'center')
s.takeaway('Trial A không tiếp tục từ best KB1 và không nối trọng số từ trial trước.')

s=new('KB3-A — Local HPO hai chiều quanh KB1','KB3  /  THIẾT KẾ A',ORANGE)
s.table(['Tham số','Anchor','Search space'],[['lr0','0.01','0.005–0.015; log scale'],['Weight decay','0.0005','0.00025–0.001; log scale'],['Momentum','0.937','Giữ nguyên'],['Augmentation strength','0.5','Giữ nguyên']],width=1050,rowh=76,colwidths=[1.4,.7,1.8],size=24)
s.text('8 trial mục tiêu\n\n0: anchor\n1: startup random\n2–7: TPE\n\nDetector seed 0',1160,253,375,440,27,ORANGE,True)
s.text('Objective: canonical validation mAP50–95 của checkpoint do native chọn.',64,679,1472,62,26)
s.takeaway('Tám trial cùng detector seed không phải tám seed xác nhận độc lập.')

s=new('KB3-A — Pretrained pilot chưa vượt baseline','KB3  /  KẾT QUẢ SNAPSHOT',ORANGE)
s.text('Snapshot: 15/10/2026, 02:31 +07:00  ·  Validation  ·  2/8 trial hoàn tất',64,214,1472,43,24,ORANGE,True)
s.table(['Mốc','mAP50–95 (%)','Δ anchor (đpt)','Δ KB1 (đpt)'],[['Supervised KB1','37.8207','+1.7530','0'],['Trial 0 — anchor','36.0677','0','−1.7530'],['Trial 1 — random','37.7225','+1.6548','−0.0982']],y=282,rowh=84,colwidths=[1.6,1.1,1.1,1],size=25)
s.text('Trial 2 đang chạy tại snapshot; chưa có completed canonical result.\nChi phí 2 trial hoàn tất: 300 epoch / 4.6795 giờ; chưa là cost cả search.',64,655,1472,92,25)
s.takeaway('Kết quả tốt nhất hoàn tất là startup random; chưa chứng minh TPE pretrained tìm tốt hơn.')

s=new('KB3-B — Quan sát, hành động và reward','KB3  /  CONTROLLER RL',ORANGE)
for x,label,body in [(64,'OBSERVATION','20 chiều quality B:\n\nTiến độ, loss, xu hướng, metric, best AP, độ trì trệ, lịch và HP.'),(563,'ACTION','LR, weight decay, momentum, augmentation strength.\n\nMask lựa chọn vượt bounds.'),(1062,'REWARD','rt = 100(bt+1 − bt)\nγ = 1\n\nbt: best canonical validation AP của epoch đã train.')]:s.card(x,235,474,470,label,body,25)
s.text('Continuity: weights, optimizer, scheduler, EMA, RNG và dataloader tiếp tục trong episode.',64,717,1472,45,22,MUTED)
s.takeaway('Quality B đã có trong code; chưa đủ bằng chứng PPO học policy hữu ích trên dữ liệu thật.')

s=two('KB3 — Kết quả kỹ thuật và định hướng',('Đã đạt','Pipeline A pretrained; tách recipe / init / reference.\n\nManifest/hash, checkpoint selection, completed-trial resume và cost.\n\nB có integration gate continuity/action.'),('Tiếp theo — theo thứ tự','1. Audit anchor và baseline; hoàn tất A.\n\n2. Xác nhận seed mới; so random cùng budget.\n\n3. B LR-only residual quanh strong A.\n\n4. Hold-only / random / progress-only; frozen-policy evaluation.'),'KB3  /  TIẾP THEO',ORANGE,'Chưa kết luận A vượt KB1, TPE vượt random hoặc B vượt A.')

s=new('Tổng hợp tiến độ ba kịch bản','TIẾN ĐỘ  /  TỔNG HỢP',GREEN)
s.table(['Hạng mục','Bằng chứng hiện có','Việc tiếp theo'],[['Dữ liệu','3.129 ảnh; 28 class; source-ID audit','Visualization; near-duplicate'],['KB1','Validation + test; G>S0 ở 21/21 cặp','Nhiều S0; bất định/objective'],['KB2','Validation; AP hơn native 3/3 seed','Holdout/test; BCE; coverage'],['KB3-A','2/8 trial ở snapshot 15/10','Audit; hoàn tất và xác nhận'],['KB3-B','Cơ chế + integration gates','Learning pilot; frozen evaluation']],rowh=78,colwidths=[.8,2.1,1.7],size=24)
s.takeaway('Đây là bảng tiến độ, không phải xếp hạng hiệu năng giữa các KB.')

s=new('Website nhận diện — kế hoạch tích hợp','ỨNG DỤNG  /  ĐỀ XUẤT',GREEN)
s.text('Ba report chưa xác nhận tiến độ triển khai website.',64,215,1472,50,27,GREEN,True)
s.flow(['Upload ảnh','API kiểm ảnh','YOLO inference','Box / class / score','Hiển thị kết quả'],y=307,height=132,size=24)
s.card(64,494,712,228,'Giao diện','Upload; xem box/class/confidence; xử lý không có detection.',24)
s.card(800,494,736,228,'Tích hợp model','Khóa checkpoint/version; đúng preprocessing; đo latency và bộ nhớ.',24)
s.takeaway('Model triển khai cần chất lượng đã xác nhận và chi phí phục vụ phù hợp.')

s=new('Kế hoạch giai đoạn tiếp theo','LỘ TRÌNH  /  TRAO ĐỔI',GREEN)
for y,label,body in [(230,'DỮ LIỆU & BÁO CÁO','Hoàn thiện visualization; thống nhất bảng số và phạm vi kết luận.'),(359,'XÁC NHẬN MÔ HÌNH','KB2 holdout/test; KB3 audit và xác nhận A; bất định KB1.'),(488,'NGHIÊN CỨU RL','Reward/FN surrogate; pilot controller B có đối chứng.'),(617,'ỨNG DỤNG','Đóng gói inference, dựng website và đo chất lượng phục vụ.')]:
 s.rect(64,y,1472,108,'FFFFFF','DCE5E1',12);s.text(label,86,y+20,405,70,25,GREEN,True);s.text(body,510,y+20,1000,82,26)
s.takeaway('CẢM ƠN  ·  TRAO ĐỔI')

# Four optional appendices, each with editable content.
s=new('A1 — Chọn checkpoint và diễn biến KB1','PHỤ LỤC  /  KB1',BLUE)
s.bullets(['Continuation: chọn best theo canonical validation, có early stopping.','Ablation: so last tại cùng 10.000 micro-step / 5.000 update.','Không gộp best-validation và fixed-final thành một mean/SD.','Các seed continuation cùng một S0; chưa phải nhiều supervised initialization.'],64,255,1472,30,32)
s.takeaway('Đường validation dùng lựa chọn checkpoint; không gọi đó là đường test.')

s=new('A2 — Từ whole-action sang conditional DPO','PHỤ LỤC  /  KB2',PURPLE)
s.flow(['Whole-action\nkhác nhiều slot','Margin tốt\nchưa chắc cứu FN','Audit score / GT','Một candidate\ngate toàn ảnh'],y=270,height=180,size=25)
s.text('Mục tiêu: gắn tín hiệu preference với quyết định có thể trace.\nRecipe lịch sử khác action space, mask và numerics; không phải ablation chỉ đổi một biến.',64,551,1472,132,29)
s.takeaway('Preference loss giảm chưa đủ; phải kiểm ordinary prediction và GT recovered/lost.')

s=new('A3 — KB2 tại cùng ngân sách FP','PHỤ LỤC  / KB2',PURPLE)
s.table(['Tại FP=984','S0 FN','Native best FN','DPO best FN'],[['Mean 3 seed','866','861.00','857.33']],y=280,rowh=95,colwidths=[1.3,.8,1.2,1.2],size=28)
s.text('DPO giảm 3.67 FN so native best tại điểm FP này.\nCutoff được quét riêng trên validation; không thay bảng chính ở confidence 0.25.',64,558,1472,133,29)
s.takeaway('Diagnostic tại một điểm; chưa chứng minh toàn PR curve vượt trội hoặc cutoff tốt trên test.')

s=new('A4 — Scratch lịch sử không gộp pretrained','PHỤ LỤC  /  KB3',ORANGE)
s.table(['Protocol','Mốc','Val mAP50–95 (%)'],[['Scratch v1','Anchor','34.8556'],['Scratch v1','Winner trial 5','36.1185'],['Pretrained KB1','Reference','37.8207']],y=255,rowh=85,colwidths=[1.2,1.2,1],size=27)
s.text('Scratch winner: +1.2629 đpt so scratch anchor; −1.7022 đpt so KB1.\nTám selected weights scratch đã xóa; logs/metrics còn lưu.',64,647,1472,94,26)
s.takeaway('Đây là pilot scratch một seed; không phải kết quả pretrained hoặc thành tích RL.')

# Add original speaker notes and precise source material.
script=(ROOT/'KICH_BAN_SLIDE_BAO_CAO_TIEN_DO_20261103.md').read_text()
blocks=re.split(r'^## Slide \d+ —.*$',script,flags=re.M)[1:]
for i,s in enumerate(slides):
 if i<32:
  block=blocks[i].split('\n---')[0]
  note=re.search(r'\*\*Lời trình bày[^*]*\*\*:?\s*(.*)',block,re.S)
  s.notes=(note.group(1).strip() if note else block)+ '\n\nĐơn vị: AP/P/R/F1 dùng %; delta dùng điểm phần trăm. SD giữa seed không phải CI.'
 else:
  key=f'A{i-31}'
  b=re.search(r'^### '+key+r' —.*?(?=^### |^## |\Z)',script,re.M|re.S)
  s.notes=b.group(0) if b else ''

# Output native editable Office Open XML, with text and vector shapes.
EMU=7620
ns='http://schemas.openxmlformats.org/drawingml/2006/main'
rels='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
def xf(it):return f'<a:xfrm><a:off x="{round(it["x"]*EMU)}" y="{round(it["y"]*EMU)}"/><a:ext cx="{round(it["w"]*EMU)}" cy="{round(it["h"]*EMU)}"/></a:xfrm>'
def fill(c):return f'<a:solidFill><a:srgbClr val="{c}"/></a:solidFill>'
def shape_xml(it,i):
 head=f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Shape {i}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
 if it['t']=='rect':
  geo='roundRect' if it.get('r') else 'rect'
  line=fill(it['line']) if it.get('line') else '<a:noFill/>'
  return head+f'<p:spPr>{xf(it)}<a:prstGeom prst="{geo}"><a:avLst/></a:prstGeom>{fill(it["fill"])}<a:ln w="9525">{line}</a:ln></p:spPr></p:sp>'
 props=f'<a:rPr lang="vi-VN" sz="{int(it["size"]*75)}" b="{int(it["bold"])}">{fill(it["color"])}<a:latin typeface="DejaVu Sans"/><a:ea typeface="DejaVu Sans"/><a:cs typeface="DejaVu Sans"/></a:rPr>'
 align={'left':'l','center':'ctr','right':'r'}[it['align']]
 paras=''.join(f'<a:p><a:pPr algn="{align}"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc><a:spcAft><a:spcPts val="0"/></a:spcAft></a:pPr><a:r>{props}<a:t>{esc(line)}</a:t></a:r></a:p>' for line in it['text'].split('\n'))
 return head+f'<p:spPr>{xf(it)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"/><a:lstStyle/>{paras}</p:txBody></p:sp>'
group='<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
def rel_xml(entries):return '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="{rels}/{t}" Target="{v}"/>' for i,t,v in entries)+'</Relationships>'
files={}
files['_rels/.rels']=rel_xml([('rId1','officeDocument','ppt/presentation.xml')])
files['ppt/presentation.xml']=f'<p:presentation xmlns:a="{ns}" xmlns:r="{rels}" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="rId2"/></p:notesMasterIdLst><p:sldIdLst>'+''.join(f'<p:sldId id="{256+i}" r:id="rId{i+3}"/>' for i in range(len(slides)))+f'</p:sldIdLst><p:sldSz cx="{W*EMU}" cy="{H*EMU}" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
files['ppt/_rels/presentation.xml.rels']=rel_xml([('rId1','slideMaster','slideMasters/slideMaster1.xml'),('rId2','notesMaster','notesMasters/notesMaster1.xml')]+[(f'rId{i+3}','slide',f'slides/slide{i+1}.xml') for i in range(len(slides))])
pns=f'xmlns:a="{ns}" xmlns:r="{rels}" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
files['ppt/slideMasters/slideMaster1.xml']=f'<p:sldMaster {pns}><p:cSld><p:spTree>{group}</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=rel_xml([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')])
files['ppt/slideLayouts/slideLayout1.xml']=f'<p:sldLayout {pns} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{group}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=rel_xml([('rId1','slideMaster','../slideMasters/slideMaster1.xml')])
files['ppt/notesMasters/notesMaster1.xml']=f'<p:notesMaster {pns}><p:cSld><p:spTree>{group}</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:notesStyle/></p:notesMaster>'
files['ppt/notesMasters/_rels/notesMaster1.xml.rels']=rel_xml([('rId1','theme','../theme/theme1.xml')])
colors={'dk1':INK,'lt1':'FFFFFF','dk2':MUTED,'lt2':BG,'accent1':GREEN,'accent2':BLUE,'accent3':PURPLE,'accent4':ORANGE,'accent5':'ABDCCB','accent6':'C4D8ED','hlink':BLUE,'folHlink':PURPLE}
files['ppt/theme/theme1.xml']=f'<a:theme xmlns:a="{ns}" name="RL YOLO"><a:themeElements><a:clrScheme name="Research">'+''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k,v in colors.items())+'</a:clrScheme><a:fontScheme name="DejaVu"><a:majorFont><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="DejaVu Sans"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Minimal"><a:fillStyleLst>'+fill('FFFFFF')*3+'</a:fillStyleLst><a:lnStyleLst>'+('<a:ln w="9525">'+fill(INK)+'</a:ln>')*3+'</a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst>'+fill('FFFFFF')*3+'</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'
for i,s in enumerate(slides,1):
 files[f'ppt/slides/slide{i}.xml']=f'<p:sld {pns}><p:cSld name="{esc(s.title)}"><p:spTree>{group}'+''.join(shape_xml(it,j+2) for j,it in enumerate(s.items))+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
 files[f'ppt/slides/_rels/slide{i}.xml.rels']=rel_xml([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','notesSlide',f'../notesSlides/notesSlide{i}.xml')])
 paras=''.join(f'<a:p><a:r><a:t>{esc(t)}</a:t></a:r></a:p>' for t in s.notes.split('\n'))
 files[f'ppt/notesSlides/notesSlide{i}.xml']=f'<p:notes {pns}><p:cSld><p:spTree>{group}<p:sp><p:nvSpPr><p:cNvPr id="2" name="Notes"/><p:cNvSpPr/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{paras}</p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'
 files[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=rel_xml([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')])
ct={'ppt/presentation.xml':'presentation.main','ppt/slideMasters/slideMaster1.xml':'slideMaster','ppt/slideLayouts/slideLayout1.xml':'slideLayout','ppt/notesMasters/notesMaster1.xml':'notesMaster','ppt/theme/theme1.xml':'theme'}
for i in range(1,len(slides)+1):ct[f'ppt/slides/slide{i}.xml']='slide';ct[f'ppt/notesSlides/notesSlide{i}.xml']='notesSlide'
files['[Content_Types].xml']='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'+''.join(f'<Override PartName="/{p}" ContentType="application/vnd.openxmlformats-officedocument.{("theme" if t=="theme" else "presentationml."+t)}+xml"/>' for p,t in ct.items())+'</Types>'
with zipfile.ZipFile(OUT/'BAO_CAO_TIEN_DO_RL_YOLO_20261103.pptx','w',zipfile.ZIP_DEFLATED) as z:
 for p,data in files.items():z.writestr(p,data)

# Browser preview / PDF source uses the exact same wrapped layout.
parts=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>RL-YOLO — Báo cáo tiến độ</title><style>@page{size:1600px 900px;margin:0}*{box-sizing:border-box}body{margin:0;background:#d9e0dc;font-family:"DejaVu Sans",sans-serif}.slide{width:1600px;height:900px;position:relative;overflow:hidden;break-after:page;margin:0 auto}.el{position:absolute;white-space:pre;line-height:1.33}@media screen{.slide{margin:24px auto;box-shadow:0 8px 35px #0002}}</style><body>']
for s in slides:
 parts.append('<section class="slide">')
 for it in s.items:
  style=f'left:{it["x"]}px;top:{it["y"]}px;width:{it["w"]}px;height:{it["h"]}px;'
  if it['t']=='rect':style+=f'background:#{it["fill"]};border-radius:{it.get("r",0)}px;'+(f'border:1px solid #{it["line"]};' if it.get('line') else '');txt=''
  else:style+=f'font-size:{it["size"]}px;color:#{it["color"]};font-weight:{700 if it["bold"] else 400};text-align:{it["align"]};';txt=esc(it['text'])
  parts.append(f'<div class="el" style="{style}">{txt}</div>')
 parts.append('</section>')
parts.append('</body></html>')
(OUT/'BAO_CAO_TIEN_DO_RL_YOLO_20261103.html').write_text('\n'.join(parts))
print(f'Built {len(slides)} slides (32 main + 4 appendices), editable PPTX with notes, matching HTML.')
