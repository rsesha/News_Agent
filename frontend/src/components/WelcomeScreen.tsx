import { InputForm } from "./InputForm";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
}) => (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-5xl mx-auto gap-4">
    <div>
      <div className="flex justify-center mb-6 animate-fadeIn">
        <img 
          src="news_hound_logo.png" 
          alt="News Hound Logo" 
          className="w-full max-w-3xl h-auto drop-shadow-[0_0_30px_rgba(255,216,0,0.4)] rounded-2xl" 
        />
      </div>
      <h1 className="text-6xl md:text-8xl font-black text-neutral-100 mb-4 tracking-tighter uppercase italic">
        News <span className="text-[var(--primary)]">Hound</span>
      </h1>
      <p className="text-lg md:text-xl lg:text-2xl text-neutral-100 font-bold w-full max-w-none mx-auto leading-relaxed drop-shadow-sm italic whitespace-nowrap">
        One Search. Multiple Sources. One Final Answer. <span className="text-[var(--primary)] uppercase">Zero Hallucinations.</span>
      </p>
    </div>
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={false}
      />
    </div>
    <p className="text-xs text-neutral-500">
      Powered by Google Gemini and LangChain LangGraph.
    </p>
  </div>
);
