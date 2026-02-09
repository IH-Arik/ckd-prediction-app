[app]

# (str) Title of your application
title = CKD Prediction App

# (str) Package name
package.name = ckdprediction

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,json,txt,pkl

# (str) Application versioning (method 1)
version = 0.1

# (list) Application requirements
requirements = python3,kivy,numpy,pandas,joblib,scikit-learn,imbalanced-learn

# (list) Supported orientations
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

#
# Android specific
#

# (list) Permissions
android.permissions = INTERNET

# (int) Target Android API, should be as high as possible.
android.api = 31

# (int) Minimum API your APK will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (bool) If True, then skip trying to update the Android sdk
android.skip_update = False

# (bool) If True, then automatically accept SDK license
android.accept_sdk_license = True

# (str) The Android arch to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.arch = armeabi-v7a

# (bool) Indicate if the application should be debuggable or not
android.debug = True

# (list) Gradle dependencies to add (can be .jar or directory)
android.gradle_dependencies =

# (list) Java classpath to add (can be .jar or directory)
android.add_jars =

# (list) Android java classpath to add (can be .jar or directory)
android.add_src =

# (str) Android additional libs to copy (comma separated)
android.add_libs_armeabi =
android.add_libs_armeabi_v7a =
android.add_libs_arm64_v8a =
android.add_libs_x86 =
android.add_libs_x86_64 =

#
# iOS specific
#

# (str) Path to a custom kivy-ios directory
ios.kivy_ios_dir = ../kivy-ios

# (str) Forces to use a specific version of kivy-ios
ios.kivy_ios_version =

# (bool) Whether to sign the ipa
ios.sign_ipa = False

#
# macOS specific
#

# (str) Path to a custom kivy-macos directory
macos.kivy_macos_dir = ../kivy-macos

#
# Windows specific
#

# (str) Path to a custom kivy-windows directory
windows.kivy_windows_dir = ../kivy-windows
