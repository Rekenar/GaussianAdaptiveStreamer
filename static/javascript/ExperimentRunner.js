class ExperimentRunner {

    constructor(baseName, networkName, events, runsPerAbr, autoExport) {
        this.baseName = baseName;
        this.networkName = networkName;
        this.events = events;
        this.runsPerAbr = runsPerAbr;
        this.autoExport = autoExport;

        this.abrOrder = ['simple', 'l2a', 'lol+'];

        this.currentRun = null;
        this.cancelled = false;
    }

    cancel() {
        this.cancelled = true;
        if (this.currentRun) this.currentRun.cancel();
    }

    async start() {

        const totalRuns = this.abrOrder.length * this.runsPerAbr;
        let doneRuns = 0;

        setExpUI({ running: true, status: "Running", progress01: 0 });

        abrToggle.checked = true;

        for (const abrKind of this.abrOrder) {
            for (let r = 1; r <= this.runsPerAbr; r++) {

                if (this.cancelled) throw new Error("CANCELLED");

                setExpUI({
                    detail: `ABR=${abrKind} run ${r}/${this.runsPerAbr}`,
                    progress01: doneRuns / totalRuns
                });

                this.currentRun = new ExperimentRun(
                    abrKind,
                    r,
                    this.baseName,
                    this.events,
                    this.networkName
                );

                await this.currentRun.start();

                doneRuns++;

                setExpUI({
                    progress01: doneRuns / totalRuns
                });

                await new Promise(r => setTimeout(r, 150));
            }
        }

        setExpUI({
            running: false,
            status: "Finished",
            progress01: 1
        });

        if (this.autoExport) {
            window.open(
                `/export?fileName=${encodeURIComponent(this.baseName)}`,
                "_blank"
            );
        }
    }
}