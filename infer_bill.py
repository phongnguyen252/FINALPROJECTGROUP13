import os
import json
import platform
import random
import string
import requests
from io import BytesIO
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.graphics.barcode import createBarcodeDrawing

class BillGenerator:
    """Lớp để tạo hóa đơn PDF từ kết quả phân loại món ăn"""
    _font_registered = False

    def __init__(self, menu_path=None, logo_path=None, vn_labels_path=None):
        '''Khởi tạo BillGenerator'''
        # Thư mục gốc dự án (parent của file này)
        base_dir = Path(__file__).parent.parent

        # Paths mặc định (vẫn cho phép truyền vào)
        self.menu_path = Path(menu_path or (base_dir / "menu.json"))
        self.logo_path = Path(logo_path or (base_dir / "logo.jpg"))
        self.vn_labels_path = Path(vn_labels_path or (base_dir / "VN_labels.json"))

        # Load menu với giá tiền (bail out nếu không có)
        with open(self.menu_path, "r", encoding="utf-8") as f:
            self.menu = json.load(f)

        # Load mapping tên -> tên tiếng Việt (nếu file tồn tại)
        if self.vn_labels_path.exists():
            try:
                with open(self.vn_labels_path, "r", encoding="utf-8") as vf:
                    # file expected: {"class_key1": "Tên tiếng Việt 1", ...}
                    self.vn_labels = json.load(vf)
            except Exception:
                # fallback empty dict nếu đọc lỗi
                self.vn_labels = {}
        else:
            self.vn_labels = {}

        # Đăng ký font hỗ trợ tiếng Việt
        self._register_vietnamese_font()


    @classmethod
    def _register_vietnamese_font(cls):
        '''Đăng ký font hỗ trợ tiếng Việt'''
        if cls._font_registered:
            return
        system = platform.system()
        font_paths = []
        if system == 'Windows':
            font_paths.extend(['C:/Windows/Fonts/arial.ttf',
                               'C:/Windows/Fonts/arialbd.ttf',
                               'C:/Windows/Fonts/verdana.ttf',
                               'C:/Windows/Fonts/verdana.ttf',])
        regular_font = None
        bold_font = None
        italic_font = None

        for font_path in font_paths:
            try:
                if not os.path.exists(font_path):
                    continue
                name = Path(font_path).stem.lower()
                if 'bold' in name or 'bd' in name:
                    if bold_font is None:
                        pdfmetrics.registerFont(TTFont('VietnameseBold', font_path))
                        bold_font = 'VietnameseBold'
                elif 'italic' in name or 'oblique' in name or 'it' in name:
                    if italic_font is None:
                        pdfmetrics.registerFont(TTFont('VietnameseItalic', font_path))
                        italic_font = 'VietnameseItalic'
                else:
                    if regular_font is None:
                        pdfmetrics.registerFont(TTFont('Vietnamese', font_path))
                        regular_font = 'Vietnamese'
                if regular_font and bold_font:
                    break
            except Exception:
                continue

        if regular_font is None:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont('Helvetica'))
                regular_font = 'Helvetica'
            except Exception:
                regular_font = 'Helvetica'
        if bold_font is None:
            try:
                pdfmetrics.registerFont(UnicodeCIDFont('Helvetica-Bold'))
                bold_font = 'Helvetica-Bold'
            except Exception:
                bold_font = regular_font
        if italic_font is None:
            italic_font = regular_font

        cls._vietnamese_font = regular_font
        cls._vietnamese_bold_font = bold_font
        cls._vietnamese_italic_font = italic_font
        cls._font_registered = True

    def calculate_bill(self, predictions):
        '''Tính hóa đơn từ kết quả phân loại'''
        items = []
        total = 0

        for pred in predictions:
            if pred is None:
                continue

            # Lấy tên lớp gốc (key) — đảm bảo tương thích với nhiều format đầu vào
            class_name = pred.get('predicted_class') or pred.get('class') or pred.get('label')
            # Tên tiếng Việt nếu pred đã chứa; nếu không sẽ dùng class_name
            vn_label = pred.get('vn_label') or pred.get('name') or class_name
            # Nếu có mapping VN, ưu tiên ghi đè bằng mapping
            if hasattr(self, 'vn_labels') and isinstance(self.vn_labels, dict):
                vn_from_map = self.vn_labels.get(class_name)
                if vn_from_map:
                    vn_label = vn_from_map

            confidence = float(pred.get('confidence') or pred.get('probability') or 0.0)
            price = int(self.menu.get(class_name, 0))

            items.append({
                'class': class_name,
                'name': vn_label,
                'price': price,
                'quantity': 1,
                'confidence': confidence
            })
            total += price

        return {
            'items': items,
            'total': total,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def generate_pdf(self, bill, output_path=None):
        '''Tạo file PDF hóa đơn'''
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(__file__).parent.parent / 'bills'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'bill_{timestamp}.pdf'

        receipt_width, receipt_height = 50*mm, 150*mm

        doc = SimpleDocTemplate(str(output_path), pagesize=(receipt_width, receipt_height),
                                rightMargin=2*mm, leftMargin=2*mm, topMargin=3*mm, bottomMargin=3*mm)
        story = []

        # Styles
        vn_font = self._vietnamese_font
        vn_bold_font = self._vietnamese_bold_font

        info_style = ParagraphStyle('info_style', fontName=vn_font, fontSize=4, leading=6)
        title_style = ParagraphStyle('ReceiptTitle', fontName=vn_bold_font, fontSize=8,
                                     spaceAfter=2*mm, alignment=1, leading=9)
        normal_style = ParagraphStyle('ReceiptNormal', fontName=vn_font, fontSize=6,
                                     spaceAfter=0.1*mm, leading=6)
        bold_style = ParagraphStyle('ReceiptBold', fontName=vn_bold_font, fontSize=7,
                                     spaceAfter=0.5*mm, leading=8)
        small_style = ParagraphStyle('ReceiptSmall', fontName=vn_font, fontSize=5,
                                     spaceAfter=0.5*mm, leading=6)

        # Logo và info
        logo_path = Path(self.logo_path)
        usable_width = receipt_width - doc.leftMargin - doc.rightMargin

        logo_width, logo_height = 11*mm, 7*mm
        logo_img = Image(str(logo_path), width=logo_width, height=logo_height)

        addr_lines = ["UEH-B2, 279 Nguyễn Tri Phương, phường", 
                      "Diên Hồng, TP. Hồ Chí Minh",
                      "Điện thoại: 034.794.5625 (Nguyễn Bá Phong)",
                      "Giờ phục vụ: 06:30 - 21:00"]
        
        info_flowables = [Paragraph(line, info_style) for line in addr_lines]

        logo_col_w = logo_width + 2*mm
        info_col_w = max(usable_width - logo_col_w, usable_width*0.4)

        header_table = Table([[logo_img, info_flowables]], colWidths=[logo_col_w, info_col_w], hAlign='LEFT')
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),]))

        story.append(header_table)
        story.append(Spacer(1, 1*mm))

        # Tiêu đề hóa đơn
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("PHIẾU THANH TOÁN", title_style))
        story.append(Paragraph("."*85, small_style))
        story.append(Spacer(1, 0.5*mm))

        # Thông tin hóa đơn
        random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        timestamp = bill.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        story.append(Paragraph(f"Thu ngân:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2025POS01", small_style))
        story.append(Paragraph(f"Thời gian:  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{timestamp}", small_style))
        story.append(Paragraph(f"Mã hóa đơn: &nbsp;&nbsp; {random_id}", small_style))
        story.append(Paragraph(f"Số món:     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{len(bill.get('items', []))}", small_style))
        story.append(Paragraph("."*85, small_style))
        story.append(Spacer(1, 1*mm))

        # Danh sách món ăn
        story.append(Paragraph("<b>DANH SÁCH MÓN ĂN</b>", bold_style))
        story.append(Spacer(1, 0.5*mm))

        table_data = []
        col_w_name = usable_width * 0.7
        col_w_price = usable_width * 0.2

        for idx, item in enumerate(bill.get('items', []), 1):
            name = f"{idx}. {item.get('name', '')}"
            price_str = f"{item.get('price', 0):,}"

            left_par = Paragraph(name, normal_style)
            right_par = Paragraph(f'<para align="right"><b>{price_str}</b></para>', normal_style)
            table_data.append([left_par, right_par])

            conf = item.get('confidence', 0.0)
            conf_text = f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Độ chính xác: {conf:.1%})"
            conf_par = Paragraph(conf_text, small_style)
            table_data.append([conf_par, ''])

        table = Table(table_data, colWidths=[col_w_name, col_w_price], hAlign='LEFT')
        ts = TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ])

        for r in range(1, len(table_data), 2):
            ts.add('SPAN', (0, r), (1, r))
            ts.add('ALIGN', (0, r), (0, r), 'LEFT')

        table.setStyle(ts)
        story.append(table)
        story.append(Spacer(1, 0.5*mm))

        story.append(Paragraph("."*85, small_style))
        story.append(Spacer(1, 0.5*mm))

        # Tổng tiền
        total_str = f"{bill.get('total', 0):,}"
        total_text = f"<b>Tổng cộng: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{total_str} VND</b>"
        story.append(Paragraph(total_text, bold_style))
        story.append(Paragraph("(Đã bao gồm thuế GTGT)", small_style))
        story.append(Paragraph("."*85, small_style))
        story.append(Spacer(1, 0.5*mm))

        # Sinh QR code tự động qua VietQR.io
        total_amount = float(bill.get('total', 0))
        qr_url = ("https://img.vietqr.io/image/sacombank-050134744526-compact2.png"
                  f"?amount={int(total_amount)}"
                  "&addInfo=THANH+TOAN+CHO+UEH+SMART+CANTEEN"
                  "&accountName=UEH+SMART+CANTEEN")

        response = requests.get(qr_url, timeout=15)
        response.raise_for_status()

        qr_bytes = BytesIO(response.content)
        qr_img = Image(qr_bytes, width=30*mm, height=35*mm)

        qr_table = Table([[qr_img]], colWidths=[usable_width], hAlign='CENTER')
        qr_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(Spacer(1, 1*mm))
        story.append(qr_table)
        story.append(Spacer(1, 0.5*mm))

        story.append(Paragraph("."*85, small_style))
        story.append(Spacer(1, 2*mm))

        # Lời cảm ơn
        footer_text = "Cảm ơn quý khách & Hẹn gặp lại"
        story.append(Paragraph(f'<para align="center">{footer_text}</para>', small_style))
        story.append(Spacer(1, 1*mm))

        # Barcode (Code128) để tra cứu hóa đơn
        bc = createBarcodeDrawing('Code128', value=random_id, barHeight=5*mm, humanReadable=False)
        bc_table = Table([[bc]], colWidths=[usable_width], hAlign='CENTER')
        bc_table.setStyle(TableStyle([
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('ALIGN', (0,0), (0,0), 'CENTER'),
        ]))

        story.append(Spacer(1, 1*mm))
        story.append(bc_table)
        story.append(Spacer(1, 2*mm))

        # Xuất file PDF và tự động mở
        doc.build(story)
        '''
        abs_path = os.path.abspath(output_path)
        system = platform.system()
        if system == "Windows":
            os.startfile(abs_path)
            try:
                if tmp_qr.exists():
                    tmp_qr.unlink()
            except Exception:
                pass
        '''
        return str(output_path)


    def generate_bill_from_predictions(self, predictions, output_path=None):
        '''Tạo hóa đơn PDF từ kết quả phân loại (hàm tổng hợp)'''
        bill = self.calculate_bill(predictions)
        pdf_path = self.generate_pdf(bill, output_path)
        return bill, pdf_path