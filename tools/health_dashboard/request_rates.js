(function (root) {
  function number(value) {
    return typeof value === 'number' && Number.isFinite(value)
  }

  function consecutiveCounterRate(samples, endIndex, options) {
    const clockAt = options.clock;
    const wallAt = options.wall;
    const valueAt = options.value;
    const resetAt = options.reset || (sample => sample.reset === true);
    const scale = options.scale || 1;
    const maxGap = options.maxGap;
    const staleAfter = options.staleAfter;
    if (endIndex < 1) return null;
    const observedClock = clockAt(samples[endIndex]);
    if (!number(observedClock) || resetAt(samples[endIndex])) return null;
    let end = endIndex;
    while (end > 0 && clockAt(samples[end - 1]) === observedClock) end--;
    if (end < 1) return null;
    const endSample = samples[end];
    const endClock = clockAt(endSample);
    if (!number(endClock)) return null;
    let base = end - 1;
    const previousClock = clockAt(samples[base]);
    while (base > 0 && clockAt(samples[base - 1]) === previousClock) base--;
    const startClock = clockAt(samples[base]);
    const age = wallAt(samples[endIndex]) - wallAt(endSample);
    if (age < 0 || age > staleAfter || resetAt(samples[base]) || !number(startClock) || endClock <= startClock) return null;
    const elapsed = (endClock - startClock) * scale;
    const firstValue = valueAt(samples[base]);
    const lastValue = valueAt(endSample);
    if (elapsed <= 0 || !number(firstValue) || !number(lastValue) || lastValue < firstValue) return null;
    let checkedClock = startClock;
    let previousValue = firstValue;
    for (let i = base + 1; i <= endIndex; i++) {
      const clock = clockAt(samples[i]);
      const value = valueAt(samples[i]);
      const wallGap = wallAt(samples[i]) - wallAt(samples[i - 1]);
      if (resetAt(samples[i]) || !number(clock) || clock < checkedClock || wallGap < 0 || wallGap > maxGap ||
          !number(value) || value < previousValue || (clock === previousClock && value !== previousValue)) return null;
      checkedClock = clock;
      previousValue = value;
    }
    return {rate: (lastValue - firstValue) / elapsed, elapsed, age,
      startClock, endClock}
  }

  function requestWindowRate(history, id, key, interval, alignedWindow = null) {
    let startIndex = history.length - 1;
    while (startIndex > 0 && history[startIndex - 1].items.has(id)) startIndex--;
    if (startIndex === history.length - 1 || !history[startIndex].items.has(id)) return null;
    const requestHistory = history.slice(startIndex);
    const gapLimit = Math.max(15, interval * 3);
    const measured = consecutiveCounterRate(requestHistory, requestHistory.length - 1, {
      clock: sample => sample.timestamp,
      wall: sample => sample.observedAt,
      value: sample => sample.items.get(id)?.[key],
      scale: 0.001,
      maxGap: gapLimit,
      staleAfter: gapLimit,
      reset: () => false
    });
    if (!measured) return null;
    if (alignedWindow && (measured.startClock !== alignedWindow.startClock || measured.endClock !== alignedWindow.endClock)) return null;
    return measured.rate
  }

  root.LLMXRates = {consecutiveCounterRate, requestWindowRate};
})(globalThis);
