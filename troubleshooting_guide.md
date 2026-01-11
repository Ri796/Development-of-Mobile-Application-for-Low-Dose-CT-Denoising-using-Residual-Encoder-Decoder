# App Stability & Troubleshooting Guide

To ensure your **RedCNN Denoising App** runs smoothly and doesn't crash, here is a summary of the fixes we applied and what to check in the future.

## 1. "Application Crash" (BLASTBufferQueue)
*   **Cause**: The new "Impeller" rendering engine in Flutter can have conflicts with certain Android Emulators or Drivers.
*   **The Fix (Applied)**: We permanently disabled Impeller in `AndroidManifest.xml`.
*   **Prevention**: You don't need to do anything. This fix is permanent.

## 2. "Infinite Loading" (Spinner doesn't stop)
*   **Cause**: The App cannot connect to the Python Backend.
*   **Reason**: Your Laptop's **IP Address changed**. This happens when you switch WiFi networks or restart the router.
*   **The Fix**: Update `const String apiUrl` in `lib/main.dart` with your current IPv4 address.
*   **Prevention**:
    1.  Open Terminal > type `ipconfig`.
    2.  Find **IPv4 Address** (e.g., `10.50.199.142`).
    3.  Check `lib/main.dart` line 99. If it's different, update it and restart the app.

## 3. "Log Reader Stopped" / ADB Error
*   **Cause**: Stale build files or connection glitch between Laptop and Phone.
*   **The Fix**:
    1.  Run `flutter clean` in the terminal.
    2.  Uninstall the app from the phone.
    3.  Run `flutter run` again.

## 4. "Still Showing Same Image"
*   **Cause**: The Python Backend code (`prep.py`) was changed, but the Server wasn't restarted.
*   **The Fix**: Always **Restart the Server** (Ctrl+C, then `python -m uvicorn...`) after editing any Python file.

---
**Summary Checklist for starting work:**
1.  [ ] Start Backend: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
2.  [ ] Check IP: Run `ipconfig` and ensure `main.dart` matches.
3.  [ ] Connect Phone: Ensure it's on the **Same WiFi**.
4.  [ ] Run App: `flutter run`
