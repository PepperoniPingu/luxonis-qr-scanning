import qrcode
from PIL import Image, ImageDraw

base_size = 21
size = base_size * 5
count = 10
clearance = 150
boarder = 10
outline_width = 5
sticker_length = (size + boarder * 2) * 3

out = Image.new("RGB", ((size + clearance + boarder * 2) * count - clearance, sticker_length), (255, 255, 255))

for i in range(count):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=size / base_size,
        border=0,
    )
    qr.add_data(f"E3PTG10{str(i)}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    out.paste(img, ((size + boarder * 2 + clearance) * i + boarder, boarder))
    draw = ImageDraw.Draw(out)
    draw.rectangle([((size + clearance + boarder * 2) * i, 0), ((size + clearance + boarder * 2) * i + size + boarder * 2, sticker_length)], outline="black", width=outline_width)
    #draw.rectangle([(0, 0), (50, 50)], outline="red", fill="red")

out.save("qr.png")
