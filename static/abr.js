class ABRAlgorithm {
    constructor(resolutionSelect) {
        this.resolutionSelect = resolutionSelect;
        this.resolutions = Array.from(resolutionSelect.options).map(opt => ({
            value: opt.value,
            width: parseInt(opt.value.split('x')[0]),
            height: parseInt(opt.value.split('x')[1])
        }));
        this.throughputHistory = [];
        this.maxHistory = 5;
        this.minThroughput = 100; 
        this.maxThroughput = 2000;
        this.lastRequestTime = null;
        this.smoothingFactor = 0.3;
        this.currentResolutionIndex = 0;
        this.resolutionHoldCount = 0;
        this.minHoldRequests = 3; 
        this.targetRequestTime = 100; 

        // Define resolution
        this.resolutionThresholds = [
            { resolution: '320x240', estImageSize: 11000, downgradeTime: 120 },
            { resolution: '480x360', estImageSize: 11000, downgradeTime: 120 },
            { resolution: '640x480', estImageSize: 25000, downgradeTime: 120 },
            { resolution: '800x600', estImageSize: 76000, downgradeTime: 120 },
            { resolution: '1024x768', estImageSize: 100000, downgradeTime: 120 },
            { resolution: '1280x720', estImageSize: 149000, downgradeTime: 120 },
            { resolution: '1366x768', estImageSize: 160000, downgradeTime: 120 },
            { resolution: '1600x900', estImageSize: 200000, downgradeTime: 120 },
            { resolution: '1920x1080', estImageSize: 274000, downgradeTime: 120 },
            { resolution: '2560x1440', estImageSize: 350000, downgradeTime: 120 }
        ];
        console.log('ABRAlgorithm initialized with resolutions:', this.resolutions.map(r => r.value));
        console.log('Resolution thresholds:', this.resolutionThresholds);
    }

    // Estimate throughput based on download time and image size
    measureThroughput(startTime, endTime, imageSize) {
        const duration = (endTime - startTime) / 1000;
        if (duration <= 0) {
            console.warn('Invalid duration for throughput measurement:', duration);
            return null;
        }
        const throughput = imageSize / duration / 1024; // KB/s
        this.throughputHistory.push(throughput);
        if (this.throughputHistory.length > this.maxHistory) {
            this.throughputHistory.shift();
        }
        const averageThroughput = this.getAverageThroughput();
        console.log(`Measured throughput: ${throughput.toFixed(2)} KB/s, Average: ${averageThroughput.toFixed(2)} KB/s, History: [${this.throughputHistory.map(t => t.toFixed(2)).join(', ')}]`);
        return averageThroughput;
    }

    // Calculate exponential moving average of throughput
    getAverageThroughput() {
        if (this.throughputHistory.length === 0) {
            console.warn('No throughput history available, returning null');
            return null;
        }
        let ema = this.throughputHistory[0];
        for (let i = 1; i < this.throughputHistory.length; i++) {
            ema = this.smoothingFactor * this.throughputHistory[i] + (1 - this.smoothingFactor) * ema;
        }
        const clampedEma = Math.min(Math.max(ema, this.minThroughput), this.maxThroughput);
        console.log(`Calculated EMA: ${ema.toFixed(2)} KB/s, Clamped: ${clampedEma.toFixed(2)} KB/s`);
        return clampedEma;
    }

    // Select resolution based on target request time
    selectResolution() {
        const throughput = this.getAverageThroughput() || this.minThroughput;
        let selectedIndex = this.currentResolutionIndex;
        let selected = this.resolutionThresholds[selectedIndex];

        if (this.resolutionHoldCount >= this.minHoldRequests) {
            // Check for upgrade
            for (let i = this.resolutions.length - 1; i >= 0; i--) {
                const predictedTime = (this.resolutionThresholds[i].estImageSize / 1024 / throughput) * 1000; // ms
                if (predictedTime <= this.targetRequestTime) {
                    selectedIndex = i;
                    break;
                }
            }
            // Check for downgrade
            const currentPredictedTime = (this.resolutionThresholds[this.currentResolutionIndex].estImageSize / 1024 / throughput) * 1000; // ms
            if (currentPredictedTime > this.resolutionThresholds[this.currentResolutionIndex].downgradeTime) {
                for (let i = this.currentResolutionIndex - 1; i >= 0; i--) {
                    const predictedTime = (this.resolutionThresholds[i].estImageSize / 1024 / throughput) * 1000; // ms
                    if (predictedTime <= this.targetRequestTime) {
                        selectedIndex = i;
                        break;
                    }
                }
            }
        }

        // Update resolution if changed
        if (selectedIndex !== this.currentResolutionIndex) {
            this.currentResolutionIndex = selectedIndex;
            this.resolutionHoldCount = 0;
        }
        this.resolutionHoldCount++;

        selected = this.resolutionThresholds[selectedIndex];
        this.resolutionSelect.value = selected.resolution;
        const selectedResolution = this.resolutions.find(r => r.value === selected.resolution);
        const predictedTime = (selected.estImageSize / 1024 / throughput) * 1000;
        console.log(`Selected resolution: ${selected.resolution}, Throughput: ${throughput.toFixed(2)} KB/s, Predicted time: ${predictedTime.toFixed(2)} ms, Hold count: ${this.resolutionHoldCount}`);
        return { width: selectedResolution.width, height: selectedResolution.height };
    }

    // Start measuring request time
    startRequest() {
        this.lastRequestTime = performance.now();
        console.log(`Started request at: ${this.lastRequestTime.toFixed(2)} ms`);
    }

    // End measuring and update resolution
    endRequest(imageSize) {
        if (this.lastRequestTime) {
            const endTime = performance.now();
            console.log(`Ended request at: ${endTime.toFixed(2)} ms, Image size: ${imageSize} bytes`);
            this.measureThroughput(this.lastRequestTime, endTime, imageSize);
            this.lastRequestTime = null;
            const resolution = this.selectResolution();
            console.log(`Updated resolution to: ${resolution.width}x${resolution.height}`);
            return resolution;
        }
        console.warn('No start time recorded for request');
        return null;
    }
}