class ExperimentRun {

    constructor(abrKind, runIndex, baseName, events, networkName) {
        this.abrKind = abrKind;
        this.runIndex = runIndex;
        this.baseName = baseName;
        this.events = events;
        this.networkName = networkName;
        this.cancelled = false;
    }

    cancel() {
        this.cancelled = true;
    }

    async start() {

        const fileName = `${this.baseName}__${this.abrKind}__run${this.runIndex}`;

        abrAlgoSelect.value = this.abrKind;
        abr = ABRFactory.create(this.abrKind, resolutionSelect);

        if (typeof abr.reset === "function") {
            abr.reset();
        }

        for (const ev of this.events) {

            if (this.cancelled) throw new Error("CANCELLED");

            if (ev.angle != null) angle = Number(ev.angle);
            if (ev.elevation != null) elevation = Number(ev.elevation);
            if (ev.x != null) x = Number(ev.x);
            if (ev.y != null) y = Number(ev.y);
            if (ev.z != null) z = Number(ev.z);

            positionDisplay.textContent =
                `Angle: ${angle.toFixed(0)}°, Elevation: ${elevation.toFixed(0)}°, X: ${x.toFixed(1)}, Y: ${y.toFixed(1)}, Z: ${z.toFixed(1)}`;

            await sendPosition();

            const predictedBps = abr.lastThroughputBps;
            const profileIdx = abr.profile;

            await reportPred(
                this.networkName,
                fileName,
                predictedBps,
                profileIdx,
                renderMs
            );

            await saveMovement(
                fileName,
                modelId,
                angle,
                elevation,
                x,
                y,
                z,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                profileIdx
            );
        }
    }
}