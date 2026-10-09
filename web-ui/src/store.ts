import { create } from "zustand";
import type { QueryResponse } from "./lib/types";

interface TraceEntry {
  id: string;
  question: string;
  at: number;
  response: QueryResponse;
}

interface AppStore {
  traces: TraceEntry[];
  addTrace: (question: string, response: QueryResponse) => void;
  clearTraces: () => void;
}

export const useAppStore = create<AppStore>((set) => ({
  traces: [],
  addTrace: (question, response) =>
    set((s) => ({
      traces: [
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          question,
          at: Date.now(),
          response,
        },
        ...s.traces,
      ].slice(0, 50),
    })),
  clearTraces: () => set({ traces: [] }),
}));
