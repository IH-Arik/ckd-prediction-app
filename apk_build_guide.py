#!/usr/bin/env python3
"""
CKD APK Build Guide
===================

Since Buildozer has compatibility issues, here are alternative solutions:
"""

import os
import sys

def show_build_options():
    print("🏥 CKD Mobile App Build Options")
    print("=" * 50)
    
    print("\n📱 OPTION 1: Use Online APK Builders")
    print("1. Upload your Python files to:")
    print("   - PytoApk (https://pytoapk.com/)")
    print("   - Repl.it APK Builder")
    print("   - PythonForAndroid.online")
    
    print("\n📱 OPTION 2: Use Bee Framework")
    print("1. Install: pip install bee")
    print("2. Convert your app to Bee format")
    print("3. Build APK using Bee's online service")
    
    print("\n📱 OPTION 3: Use Chaquopy (Professional)")
    print("1. Install: pip install chaquopy")
    print("2. Follow Chaquopy documentation")
    print("3. Build native Android app")
    
    print("\n📱 OPTION 4: WebView App (Easiest)")
    print("1. Create simple Android WebView app")
    print("2. Load your Streamlit app URL")
    print("3. Package as APK using Android Studio")
    
    print("\n📱 OPTION 5: Use Kivy Online Builder")
    print("1. Use Kivy's official build service")
    print("2. Upload your .py file")
    print("3. Download generated APK")
    
    print("\n🔧 QUICK FIX - Try Different Buildozer Command:")
    print("Try these commands:")
    print("  python -m buildozer android debug")
    print("  buildozer -v android debug")
    print("  buildozer android debug deploy")
    
    print("\n📋 REQUIREMENTS CHECK:")
    print("Make sure you have:")
    print("- Java JDK 8+")
    print("- Android SDK")
    print("- Android NDK")
    print("- Set ANDROID_HOME environment variable")
    
    print("\n🌐 ALTERNATIVE: Streamlit Cloud")
    print("Easiest option:")
    print("1. Push to GitHub")
    print("2. Deploy to Streamlit Cloud")
    print("3. Access via mobile browser")
    print("4. Add to home screen (works like app)")

def create_simple_webview():
    """Create a simple WebView app template"""
    
    webview_code = '''
import webview
import sys

def main():
    # Create WebView window
    window = webview.create_window(
        'CKD Prediction',
        'http://your-streamlit-app-url:8501',
        width=400,
        height=800,
        resizable=True,
        fullscreen=False
    )
    webview.start()

if __name__ == '__main__':
    main()
'''
    
    with open('ckd_webview_app.py', 'w') as f:
        f.write(webview_code)
    
    print("✓ Created 'ckd_webview_app.py'")
    print("Install: pip install pywebview")
    print("Run: python ckd_webview_app.py")
    print("Convert to APK using PytoApk or similar service")

def show_streamlit_mobile_info():
    """Show how to make Streamlit mobile-friendly"""
    
    print("\n📱 STREAMLIT MOBILE OPTIMIZATION:")
    print("=" * 50)
    
    mobile_tips = '''
# Add to your Streamlit app:
import streamlit as st

# Mobile detection
def is_mobile():
    user_agent = st.experimental_get_query_params().get('user_agent', [''])[0]
    return 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent

if is_mobile():
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 16px !important;
    }
    .stButton > button {
        font-size: 16px !important;
        padding: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Mobile-specific layout
    st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1.0">', unsafe_allow_html=True)
'''
    
    print(mobile_tips)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        option = sys.argv[1]
        
        if option == "webview":
            create_simple_webview()
        elif option == "mobile":
            show_streamlit_mobile_info()
        else:
            print("Usage: python apk_build_guide.py [webview|mobile]")
    else:
        show_build_options()
