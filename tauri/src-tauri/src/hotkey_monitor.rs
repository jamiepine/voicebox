//! Global hotkey → dictation effect bridge.
//!
//! Thin adapter from `keytap::chord::ChordMatcher` to Tauri events. keytap
//! owns the OS event tap + the chord state machine (Momentary vs Toggle,
//! longest-match resolution, sticky-toggle semantics); this module's only
//! job is:
//!
//!   1. Build a `ChordMatcher` from the user's saved PTT + Toggle chords.
//!   2. Translate `ChordEvent` → voicebox's [`Effect`] on a dispatcher
//!      thread.
//!   3. Fan [`Effect`]s out into Tauri events + dictate-window show/hide.
//!
//! The [`Effect::RestartRecording`] signal is emitted when keytap fires
//! `End(PTT)` and `Start(Toggle)` with the *same* [`Instant`] — which
//! happens when the held set upgrades from a shorter chord to a longer
//! superset in a single event (the classic PTT→hands-free transition).
//! We detect the pair with a 5 ms peek on the matcher's receiver and
//! coalesce into one `Restart` so hosts can discard the transition-
//! moment audio rather than treat it as an unrelated Stop+Start pair.
//!
//! Left- and right-hand modifier variants are kept distinct all the way
//! down to the OS event tap (keytap's core promise). Defaults bind to
//! right-hand Cmd + right-hand Option on macOS / right-hand Ctrl +
//! right-hand Shift on Windows so the usual left-hand shortcuts stay
//! with the OS / app.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use keytap::chord::{Chord, ChordEvent, ChordMatcher};
use keytap::{Key, RecvTimeoutError};
use tauri::{AppHandle, Emitter, Manager};

use crate::focus_capture;
use crate::DICTATE_WINDOW_LABEL;

// ========================================================================
// Public types
// ========================================================================

/// Semantic action a chord can be bound to. `PushToTalk` = hold chord to
/// record, release to stop. `ToggleToTalk` = press chord to start recording,
/// press again to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChordAction {
    PushToTalk,
    ToggleToTalk,
}

/// Effect produced after the chord matcher resolves an event. Hosts
/// translate these into UI / recorder calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    StartRecording(ChordAction),
    StopRecording(ChordAction),
    /// Emitted when a push-to-talk chord is "upgraded" into the toggle
    /// chord mid-hold — hosts may want to discard the captured audio and
    /// restart so the transition moment isn't in the recording.
    RestartRecording(ChordAction),
}

/// Chord key sets from capture settings. Both actions use the same
/// `HashSet<Key>` shape so callers don't need to know about keytap's
/// `Chord` type.
pub type Bindings = HashMap<ChordAction, HashSet<Key>>;

// ========================================================================
// Monitor
// ========================================================================

pub struct HotkeyMonitor {
    app: AppHandle,
    active: Option<Active>,
}

struct Active {
    dispatcher: JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
}

impl HotkeyMonitor {
    /// Build the monitor with initial bindings. Equivalent to constructing
    /// an empty monitor and calling [`Self::update_bindings`] once.
    pub fn spawn(app: AppHandle, bindings: Bindings) -> Self {
        let mut m = Self { app, active: None };
        m.apply(bindings);
        m
    }

    /// Swap in a fresh set of chord bindings. Tears down the existing
    /// `ChordMatcher` (which stops keytap's chord worker thread and
    /// closes the OS tap) and spawns a new one. No-op for the "all
    /// empty" case so "disable hotkey" doesn't keep a tap running for
    /// no reason.
    pub fn update_bindings(&mut self, bindings: Bindings) {
        self.apply(bindings);
    }

    fn apply(&mut self, bindings: Bindings) {
        // Tear down any existing matcher + dispatcher first. The
        // dispatcher sees the shutdown flag on its next recv_timeout
        // (≤100ms) and returns; joining waits for that. Dropping the
        // ChordMatcher stops keytap's chord-worker thread and the
        // underlying Tap.
        if let Some(active) = self.active.take() {
            active.shutdown.store(true, Ordering::Relaxed);
            let _ = active.dispatcher.join();
        }

        if bindings.values().all(|set| set.is_empty()) {
            return;
        }

        let matcher = match build_matcher(&bindings) {
            Ok(m) => m,
            Err(err) => {
                eprintln!(
                    "HotkeyMonitor: ChordMatcher build failed ({err}). Global chord detection is disabled. On macOS, grant Input Monitoring in System Settings → Privacy & Security → Input Monitoring and relaunch."
                );
                return;
            }
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_for_thread = shutdown.clone();
        let app = self.app.clone();
        let dispatcher = thread::Builder::new()
            .name("voicebox-hotkey-dispatcher".into())
            .spawn(move || dispatcher_loop(app, matcher, shutdown_for_thread))
            .expect("spawn hotkey dispatcher thread");

        self.active = Some(Active { dispatcher, shutdown });
    }
}

impl Drop for HotkeyMonitor {
    fn drop(&mut self) {
        if let Some(active) = self.active.take() {
            active.shutdown.store(true, Ordering::Relaxed);
            let _ = active.dispatcher.join();
        }
    }
}

// ========================================================================
// Matcher construction + dispatch
// ========================================================================

fn build_matcher(bindings: &Bindings) -> Result<ChordMatcher<ChordAction>, keytap::Error> {
    let mut builder = ChordMatcher::builder();
    if let Some(keys) = bindings.get(&ChordAction::PushToTalk) {
        if !keys.is_empty() {
            builder = builder.add(
                ChordAction::PushToTalk,
                Chord::of(keys.iter().copied()),
            );
        }
    }
    if let Some(keys) = bindings.get(&ChordAction::ToggleToTalk) {
        if !keys.is_empty() {
            builder = builder.add_toggle(
                ChordAction::ToggleToTalk,
                Chord::of(keys.iter().copied()),
            );
        }
    }
    builder.build()
}

fn dispatcher_loop(
    app: AppHandle,
    matcher: ChordMatcher<ChordAction>,
    shutdown: Arc<AtomicBool>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        match matcher.recv_timeout(Duration::from_millis(100)) {
            Ok(event) => process_event(&app, &matcher, event),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Turn a single [`ChordEvent`] into zero or one [`Effect`]s, peeking at
/// the matcher once for a same-Instant follow-up so upgrade transitions
/// coalesce into [`Effect::RestartRecording`] instead of a Stop+Start
/// pair.
fn process_event(
    app: &AppHandle,
    matcher: &ChordMatcher<ChordAction>,
    event: ChordEvent<ChordAction>,
) {
    match event {
        ChordEvent::Start { id, .. } => {
            apply_effect(app, Effect::StartRecording(id));
        }
        ChordEvent::End { id: end_id, time: end_time } => {
            // Peek for an immediately-following Start. keytap emits
            // End+Start atomically (same Instant) when the held set
            // transitions between registered chords — our 5 ms window
            // is well under perceptible latency but far longer than the
            // channel hop between keytap's chord worker and our
            // dispatcher.
            match matcher.recv_timeout(Duration::from_millis(5)) {
                Ok(ChordEvent::Start { id: start_id, time: start_time })
                    if start_time == end_time =>
                {
                    apply_effect(app, Effect::RestartRecording(start_id));
                }
                Ok(other) => {
                    apply_effect(app, Effect::StopRecording(end_id));
                    // The peeked event wasn't a transition partner;
                    // process it in its own right. Recursion depth is
                    // bounded by the number of back-to-back chord
                    // events, in practice 1–2.
                    process_event(app, matcher, other);
                }
                Err(_) => {
                    apply_effect(app, Effect::StopRecording(end_id));
                }
            }
        }
    }
}

// ========================================================================
// Effect → Tauri
// ========================================================================

fn apply_effect(app: &AppHandle, effect: Effect) {
    match effect {
        Effect::StartRecording(_) => {
            // Snapshot focus BEFORE we touch the window — any AppKit
            // reshuffle triggered by set_position / show could in principle
            // steal key focus and poison the reading. In practice those
            // calls leave keyWindow alone, but capturing first is free.
            let focus = focus_capture::capture_focus().ok();

            if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                // The previous hide-cycle parked the window off-screen and
                // made it click-through — undo both before showing, so the
                // pill lands at top-center and the user can actually click
                // the error pill / stop button.
                //
                // `current_monitor()` returns None when the window is off
                // any display (our hide handler parks it at -10_000, -10_000
                // precisely so it never intercepts clicks), so fall back to
                // the primary monitor for the reposition.
                let monitor = window
                    .current_monitor()
                    .ok()
                    .flatten()
                    .or_else(|| window.primary_monitor().ok().flatten());
                if let Some(monitor) = monitor {
                    let monitor_pos = monitor.position();
                    let monitor_size = monitor.size();
                    if let Ok(win_size) = window.outer_size() {
                        let x = monitor_pos.x
                            + (monitor_size.width as i32 - win_size.width as i32) / 2;
                        let y = monitor_pos.y + (monitor_size.height as f64 * 0.04) as i32;
                        let _ = window.set_position(tauri::PhysicalPosition::new(x, y));
                    }
                }
                let _ = window.set_ignore_cursor_events(false);
                // Deliberately no set_focus() — taking key focus would yank
                // it out of whatever app the user was typing in, which is
                // the opposite of what a dictation overlay should do.
                let _ = window.show();
                let payload = serde_json::json!({ "focus": focus });
                let _ = window.emit("dictate:start", payload);
            }
        }
        Effect::StopRecording(_) => {
            if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                let _ = window.emit("dictate:stop", ());
            }
        }
        Effect::RestartRecording(_) => {
            if let Some(window) = app.get_webview_window(DICTATE_WINDOW_LABEL) {
                let _ = window.emit("dictate:restart", ());
            }
        }
    }
}
