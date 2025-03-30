import streamlit as st
import joblib
import pandas as pd
import numpy as np

# بارگذاری مدل
model = joblib.load('laptop_price_predictor.pkl')

# عنوان اپلیکیشن
st.title('💻 پیش‌بینی قیمت لپ‌تاپ')

# توضیحات
st.markdown("""
این اپلیکیشن قیمت لپ‌تاپ را بر اساس ویژگی‌های آن پیش‌بینی می‌کند. لطفاً مشخصات لپ‌تاپ را وارد کنید.
""")

# تصویر یا آیکون
st.image('https://cdn-icons-png.flaticon.com/512/2965/2965300.png', width=100)

# فرم ورودی داده‌ها
st.sidebar.header('📝 مشخصات لپ‌تاپ')

brand = st.sidebar.selectbox('برند', ['HP', 'Lenovo', 'Dell', 'Asus', 'Acer'])
processor_brand = st.sidebar.selectbox('برند پردازنده', ['Intel', 'AMD', 'MediaTek'])
ram = st.sidebar.slider('حافظه RAM (GB)', 4, 32, 8)
ram_type = st.sidebar.selectbox('نوع RAM', ['DDR4', 'DDR5', 'LPDDR4'])
ghz = st.sidebar.slider('سرعت پردازنده (GHz)', 1.0, 5.0, 2.5)
ssd = st.sidebar.slider('حافظه SSD (GB)', 128, 2048, 512)
hdd = st.sidebar.selectbox('حافظه HDD', ['No HDD', '512 GB', '1 TB', '2 TB'])
gpu_brand = st.sidebar.selectbox('برند GPU', ['NVIDIA', 'Intel', 'AMD', 'Integrated'])

# تبدیل داده‌های ورودی به فرمت مدل
input_data = {
    'RAM': ram,
    'Ghz': ghz,
    'SSD': ssd,
    'Brand_' + brand: True,
    'Processor_Brand_' + processor_brand: True,
    'RAM_TYPE_' + ram_type: True,
    'HDD_' + hdd: True,
    'GPU_Brand_' + gpu_brand: True
}

# ایجاد DataFrame از داده‌های ورودی
input_df = pd.DataFrame([input_data])

# پر کردن مقادیر گم‌شده (ستون‌های دیگر را False می‌کنیم)
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = False

# مرتب‌سازی ستون‌ها مطابق مدل
input_df = input_df[model.feature_names_in_]

# پیش‌بینی قیمت
if st.sidebar.button('پیش‌بینی قیمت'):
    prediction = model.predict(input_df)
    st.success(f'💰 قیمت پیش‌بینی شده: **{int(prediction[0]):,}** تومان')

# نمایش اطلاعات اضافی
st.markdown("""
### راهنما:
- **برند**: برند لپ‌تاپ را انتخاب کنید.
- **برند پردازنده**: سازنده پردازنده را انتخاب کنید.
- **RAM**: مقدار حافظه رم را انتخاب کنید.
- **نوع RAM**: نوع حافظه رم را انتخاب کنید.
- **سرعت پردازنده**: سرعت پردازنده را انتخاب کنید.
- **SSD**: حجم حافظه SSD را انتخاب کنید.
- **HDD**: حجم حافظه HDD را انتخاب کنید.
- **برند GPU**: سازنده کارت گرافیک را انتخاب کنید.
""")
