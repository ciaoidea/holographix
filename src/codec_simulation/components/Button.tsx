import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = 'primary', 
  icon, 
  className = '', 
  ...props 
}) => {
  const baseStyles = "relative font-mono uppercase tracking-wider text-xs font-bold py-2 px-4 transition-all duration-200 flex items-center gap-2 group overflow-hidden border";
  
  const variants = {
    primary: "bg-holo-900/40 border-holo-500 text-holo-300 hover:bg-holo-500 hover:text-white hover:shadow-[0_0_15px_rgba(20,184,166,0.5)]",
    secondary: "bg-space-800 border-slate-600 text-slate-400 hover:border-slate-400 hover:text-slate-200",
    danger: "bg-red-900/20 border-red-500 text-red-400 hover:bg-red-600 hover:text-white hover:shadow-[0_0_15px_rgba(220,38,38,0.5)]",
    ghost: "border-transparent text-slate-500 hover:text-holo-400 bg-transparent"
  };

  return (
    <button className={`${baseStyles} ${variants[variant]} ${className}`} {...props}>
      {/* Corner decorations */}
      <span className="absolute top-0 left-0 w-1 h-1 border-t border-l border-current opacity-50" />
      <span className="absolute bottom-0 right-0 w-1 h-1 border-b border-r border-current opacity-50" />
      
      {icon && <span className="w-4 h-4">{icon}</span>}
      {children}
    </button>
  );
};