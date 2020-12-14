#ifndef INCLUDE_STIMER
#define INCLUDE_STIMER


class STimer {
public:
    STimer();
    ~STimer();
    void Start(void);
    void Stop(void);
    double Elapsed(void);
    void Clear(void);

    int timeron;

    struct timeval tuse, tstart, timer;
};

STimer::STimer(void) { 

    timeron = false;

    timerclear(&timer);
    timeron = 0;
}

STimer::~STimer() {
}

void STimer::Start() {
    assert(!timeron); 
    assert( gettimeofday( &(tstart), (struct timezone *)NULL ) == 0 );
    timeron = 1;
}

void STimer::Stop(void) {
    assert( timeron );
    struct timeval dt;
    assert( gettimeofday( &(tuse), (struct timezone *)NULL ) == 0 );
    timersub(&(tuse), &(tstart), &dt);
    timeradd(&dt, &(timer), &(timer));
    timeron = 0;
}

void STimer::Clear(void) {
    assert(!timeron); 
    timerclear(&(timer));
}

double STimer::Elapsed(void) {
    return  timer.tv_sec + 1e-6*timer.tv_usec;
}

#endif // INCLUDE_STIMER
