## Auto-Start on Boot

To configure the script to run automatically when your Raspberry Pi starts:

### 1. Create the systemd service file
```bash
sudo nano /etc/systemd/system/(name).service
```

### 2. Paste this configuration
```ini
[Unit]
Description=(your description here)
After=network.target

[Service]
Type=simple
User=(username)
WorkingDirectory=/home/(username)/ai_science_fair_proj
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/(username)/.Xauthority
ExecStart=/usr/bin/python3 /home/(username)/ai_science_fair_proj/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Note:** Replace `(username)` with your actual Raspberry Pi username and adjust paths if your project is in a different location.

### 3. Save and exit
Press `Ctrl+X`, then `Y`, then `Enter`

### 4. Enable and start the service
```bash
# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable (name).service

# Start the service now (without rebooting)
sudo systemctl start (name).service
```

### 5. Verify it's running
```bash
sudo systemctl status (name).service
```

### Useful Commands
```bash
# Stop the service
sudo systemctl stop (name).service

# Restart the service
sudo systemctl restart (name).service

# Disable auto-start
sudo systemctl disable (name).service

# View live logs
journalctl -u (name).service -f
```
**Note:** Replace `(name)` with your actual script name.
