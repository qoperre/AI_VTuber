import os
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

process_is_running = False
process_thread = None

def run_process(tts_type: str, line_callback: callable = None):
    def _run_process():
        global process_is_running
        process_is_running = True
        try:
            curr_env = os.environ.copy()
            curr_env["TTS_TYPE"] = tts_type
            proc = subprocess.Popen(
                ["python", "run.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                env=curr_env
            )
            while process_is_running:
                line = proc.stdout.readline()
                if not line:
                    break
                if line_callback:
                    line_callback(line)
                print(line.decode("utf-8").rstrip())
            proc.kill()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            process_is_running = False
            print("Process stopped.")

    global process_thread
    process_thread = threading.Thread(target=_run_process)
    process_thread.start()

def stop_process():
    global process_is_running, process_thread
    process_is_running = False
    if process_thread:
        process_thread.join()
        process_thread = None
        print("Stopped manually.")

class RunGUI:
    def __init__(self):
        root = tk.Tk()
        root.title("AI-Vtuber GUI")
        style = ttk.Style()
        style.theme_use("vista")
        root.geometry("400x300")

        frame = ttk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tts_label = ttk.Label(frame, text="TTS Type:")
        tts_label.grid(row=0, column=0, padx=5, pady=5)
        tts_dropdown = ttk.Combobox(frame, values=["pyttsx3", "EL"])
        tts_dropdown.current(1)
        tts_dropdown.grid(row=0, column=1, padx=5, pady=5)

        console = tk.Text(frame, height=10, width=50)
        console.grid(row=1, column=0, columnspan=2, padx=5, pady=10)

        run_btn = ttk.Button(frame, text="Run", command=lambda: run_process(tts_dropdown.get(), lambda l: console.insert(tk.END, l)))
        run_btn.grid(row=2, column=0, padx=5, pady=5)

        stop_btn = ttk.Button(frame, text="Stop", command=stop_process)
        stop_btn.grid(row=2, column=1, padx=5, pady=5)

        self.root = root

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    RunGUI().run()
