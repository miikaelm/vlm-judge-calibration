% ------------------------------
% How to use? Store the file in
% 1) Windows XP C:\Documents and Settings\<user name>\texmf\tex\latex\local\
% 2) Windows Vista/7 C:\Users\<user name>\texmf\tex\latex\local\
% 3) Linux ~/texmf/tex/latex/local/
% 4) Mac OS X /Users/<user name>/Library/texmf/tex/latex/local/
% 5) Sharelatex: just copy this file to the project folder
% 6) or in the same directory as the document
% ------------------------------

\NeedsTeXFormat{LaTeX2e}[2015/01/01]
\ProvidesClass{utuftthesis}[2023/08/02 UTU FT dept BSc/MSc thesis]

%% package requirements for this template
\RequirePackage{kvoptions}                     % keyval-options, required by this class definition
\RequirePackage{adjustbox}                     % front page box on the right
\RequirePackage{etoolbox}                      % misc. tools
\RequirePackage{ifthen}                        % mainly: \langchoose and \switch (\ifstrequal didn't work)
\RequirePackage{ifxetex}                       % xelatex/pdflatex font selection
\RequirePackage{fancyhdr}                      % custom headers/footers
\RequirePackage{xstring}                       % string comparison
\RequirePackage[lastpage,totpages,user]{zref}  % page counting logic
\RequirePackage{setspace}                      % linespacing

%% Custom package options
%\DeclareStringOption[00]{year}
\DeclareStringOption[draft]{version}           % draft/final version
\DeclareStringOption[a4paper]{paper}           % @see ctan/macros/latex/base/classes, 3.1 Setting Paper Size
\DeclareStringOption[oneside]{sides}           % oneside/twoside
\DeclareStringOption[finnish]{language}        % document's main language
\DeclareStringOption[ieee]{bibstyle}           % document's bibliography style
\DeclareStringOption[-1]{pagecountdelta}       % workaround for bugs in the app page count
\DeclareStringOption[a-2b,mathxmp]{pdfaconformance}  % PDF-A conformance level (a-1b, a-2b, a-2u, a-3b, a-3u, or 'none')
\DeclareStringOption[split]{biburlbreak}       % how to handle long urls in bibliography ('ragged', 'split', or 'none')
\DeclareStringOption[times]{mainfont}          % pdftex: main font package (lmodern, fourier, ...), special value: 'times', 'none'
                                               % xetex: main TTF font name (Arial, ...), special value: 'times' enables Times New Roman, 'none'
\DeclareBoolOption[true]{hidechapters}         % hide the verbose chapter title text
\DeclareBoolOption[true]{hidelinks}            % hide the underlining of links (PDF-A & hyperref)
\DeclareBoolOption[false]{countbibpages}       % page count = content | content + bibliography pages?
\DeclareBoolOption[false]{sharelatex}          % assume compilation inside sharelatex
\DeclareBoolOption[false]{minted}              % use minted instead of listings (must compile with '-shell-escape')
\DeclareBoolOption[true]{emptyfirstpages}      % hide the header/footer of the first page of each chapter
\DeclareBoolOption[true]{turnitinchecked}      % turnitin text
\DeclareBoolOption[false]{turnitinnewpage}     % turnitin text on a new page
\DeclareBoolOption[false]{hidetocpagenum}      % hide the roman page numbers in toc / lists


%% Process given options
\ProcessKeyvalOptions*

%% Load base
\LoadClass[\utuftthesis@paper,\utuftthesis@sides,12pt]{report}

%% !! Update this only if the document's main language is not finnish or english !!
%% The template supports one of three options
%%  * \n@tivelang + english
%%  * just \n@tivelang
%%  * just english
\def\n@tivelang{finnish}

%% macros for conditional language blocks
\newcommand{\l@nguage}{\utuftthesis@language}
\newcommand{\langchoose}[3]{\ifthenelse{\equal{#1}{\n@tivelang}}{#2}{#3}}
\newcommand{\langfin}[2]{\langchoose{\l@nguage}{#1}{#2}}

%% switch-case used to select the thesis type
\newbool{default}
\newcommand{\case}{}
\newcommand{\default}{}
\newenvironment{switch}[1]{%
    \booltrue{default}
    \renewcommand{\case}[2]{\ifthenelse{\equal{#1}{##1}}{%
        \boolfalse{default}##2}{}}%
    \renewcommand{\default}[1]{\ifbool{default}{##1}{}}
}{}

%% Set up encoding/font/hyphenation stuff
%% Note: this basic setup is mainly meant for latex newbies unaware of
%% the fine art of typography. The basic font configuration can be set
%% up with the document parameters 'realtimesnewroman' & 'mainfont'
%% Configurations:
%%  * realtimesnewroman=F on luatex/xetex: uses the standard LaTeX fonts
%%  * realtimesnewroman=F on pdftex: sets up the font package 'mainfont'
%%  * realtimesnewroman=T on pdftex: uses the default Times font family
%%    (may look similar to MS Office)
%%  * realtimesnewroman=T on luatex/xetex: uses Times New Roman or fails (must be found)
%%    (may look similar to MS Office)
%%  * advanced:
%%    set up the font settings in the document header
%%    some suggestions: opentype, xetex, microtype, ligatures,
%%    commercial fonts, mixing & matching different fonts
%%    @see https://en.wikibooks.org/wiki/LaTeX/Fonts#Using_alternative_fonts
%%    @see http://www.khirevich.com/latex/microtype/
\ifxetex
  \RequirePackage{fontspec}

  % https://bitbucket.org/martin_scharrer/filehook/issues/1/latex-will-change-definition-of
  \RequirePackage{scrlfile}

  \RequirePackage{polyglossia}
  % see bug https://github.com/reutenauer/polyglossia/issues/331
  %\setmainlanguage{\utuftthesis@language}
  \begingroup\edef\x{\endgroup\noexpand\setdefaultlanguage{\utuftthesis@language}}\x%
  \langfin{\setotherlanguage{english}}
  
  \ifdefstring{\utuftthesis@mainfont}{times}{
    \ifutuftthesis@sharelatex
      % ShareLaTeX doesn't ship with (C) Microsoft corefonts,
      % need to provide the font files locally inside the project
      \setmainfont[BoldFont=timesbd.ttf,ItalicFont=timesi.ttf,BoldItalicFont=timesbi.ttf]{times.ttf}
    \else
      \setmainfont{Times New Roman}
    \fi
  }{
    \ifdefstring{\utuftthesis@mainfont}{none}{
    }{
      \setmainfont{\utuftthesis@mainfont}
    }
  }
\else
  % JH: modified latin to UTF-8 encoding cues to make Scandinavian characters work
  \RequirePackage[T1]{fontenc}
  \RequirePackage[utf8]{inputenc}

  % load babel-package for automatic hyphenation
  \ifdefstring{\utuftthesis@language}{english}{
    \RequirePackage[\n@tivelang, english]{babel}
  }{
    \RequirePackage[english, \l@nguage]{babel}
  }
  
  \ifdefstring{\utuftthesis@mainfont}{none}{
  }{
    % Note: the system picks the configured default Times font family font
    % On Linux/Mac/ShareLaTeX, could be Nimbus or other font, not Times New Roman due to license issues
    \ifdefstring{\utuftthesis@mainfont}{times}{
      % roman font: times
      \renewcommand{\rmdefault}{ptm}
    }{
      \RequirePackage{\utuftthesis@mainfont}
    }
    
    % sans-serif: helvetica
    \renewcommand{\sfdefault}{phv}
    
    % monospace font: courier
    \renewcommand{\ttdefault}{pcr}
  }
\fi

\ifdefstring{\utuftthesis@version}{draft}{
  \newcommand{\version}{[draft]}

}{
  \newcommand{\version}{}
  
  \ifdefstring{\utuftthesis@pdfaconformance}{none}{
  }{
    %% @see http://ftp.acc.umu.se/mirror/CTAN/macros/latex/contrib/pdfx/pdfx.pdf
    %% @see https://en.wikipedia.org/wiki/PDF/A#Conformance_levels_and_versions
    \RequirePackage{colorprofiles}
    \RequirePackage[\utuftthesis@pdfaconformance]{pdfx}
    
    %% fix a temporary bug in texlive 2021
    \catcode30=12
    
    %% hides the underlining of links with red (by hyperref)
    %% @see http://mirrors.ctan.org/macros/latex/contrib/hyperref/doc/manual.pdf
    %% note: pdfx implicitly imports hyperref
    \ifutuftthesis@hidelinks
      \hypersetup{hidelinks}
    \fi
  }
}

%% not used by the class, but usually needed
%% if you encounter package clashes, start by disabling these
%% @see https://tex.stackexchange.com/a/26592
%% @see https://tex.stackexchange.com/q/553
%% @see https://tex.stackexchange.com/a/26200
%% @see https://www.ctan.org/topic/obsolete
\RequirePackage{verbatim}               % LyX generated \begin{comment}\end{comment}
\RequirePackage{graphicx}               % \includegraphics{file}, works without file extensions
\RequirePackage{amssymb,amsthm,amsmath} % ams-packages for maths
\RequirePackage{nomencl}                % acronyms, https://tex.stackexchange.com/questions/154060/nomenclature-acronym-or-glossary
                                        % https://www.overleaf.com/learn/latex/Nomenclatures

%% inclusion of source code
%% - listings is more widely supported, but does not handle utf-8
%% - minted supports utf-8, but requires external tools: Python & pygments
%% - minted also requires compiling the document with -shell-escape
\ifutuftthesis@minted
  \RequirePackage{minted}
\else
  \RequirePackage{listings}
  \newenvironment{minted}{\verbatim}{\endverbatim}
  \newcommand{\listingscaption}{\sourcecode}
\fi

%% some more suggestions
%\RequirePackage[l2tabu, orthodox]{nag} % checks whether outdated latex features are being used
%\RequirePackage[all,warning]{onlyamsmath}
%\RequirePackage{titlecaps}              % capitalized words in titles
%\RequirePackage{microtype}              % micro-typographic extensions
%\RequirePackage{fixmath}                % ISO compliant math typesetting
%\RequirePackage[euro]{isonums}          % euro style number formatting
%\RequirePackage[amsmath,thmmarks,hyperref]{ntheorem}
%\RequirePackage{filecontents}           % to output the pdfx metadata file
%\RequirePackage[section]{placeins}      % see https://en.wikibooks.org/wiki/LaTeX/Floats,_Figures_and_Captions#Keeping_floats_in_their_place

% bold acronyms in the list of acronyms
\renewcommand{\nomlabel}[1]{{\textbf{#1}}}
\makenomenclature
\newcommand{\listofacronyms}{\printnomenclature}

%% packages that depend on language packages / definitions

%% @see https://tex.stackexchange.com/a/134281
\ifdefstring{\utuftthesis@biburlbreak}{break}{
  \setcounter{biburllcpenalty}{7000}
  \setcounter{biburlucpenalty}{8000}
}{}

%% bibliography
%% - uses the biber / biblatex engine: supports utf-8 and easier formatting
\ifdefstring{\utuftthesis@biburlbreak}{ragged}{
  \RequirePackage[style=\utuftthesis@bibstyle,backend=biber,block=ragged]{biblatex}
}
{
  \RequirePackage[style=\utuftthesis@bibstyle,backend=biber]{biblatex} 
}

\RequirePackage{fvextra}                % minted
\RequirePackage{csquotes}               % quotations

% month names
\AtEndPreamble{
  \RequirePackage[useregional,calc]{datetime2}
}


%% !! Update these only if the document's main language is not finnish or english !!
\newcommand{\ye@r}{\the\year}
\newcommand{\m@nth}{\the\month}
\newcommand{\p@blab}{Labra}
\newcommand{\p@blaben}{Laboratory}
\newcommand{\s@pervisors}{none}
\newcommand{\k@ywords}{not set}
\newcommand{\k@ywordstwo}{not set}
\newcommand\keywords[1]{\renewcommand\k@ywords{#1}}
\newcommand\keywordstwo[1]{\renewcommand\k@ywordstwo{#1}}
\newcommand\keywordsen[1]{\renewcommand\k@ywordstwo{#1}}
\newcommand\pubyear[1]{\renewcommand\ye@r{#1}}
\newcommand\pubmonth[1]{\renewcommand\m@nth{#1}}
\newcommand\publab[1]{\renewcommand\p@blab{#1}}
\newcommand\publaben[1]{\renewcommand\p@blaben{#1}}
\newcommand\pubtype[1]{\newcommand\p@btype{#1}}
\newcommand\supervisors[1]{\renewcommand\s@pervisors{#1}}
\newcommand{\department}[1]{\langchoose{#1}{Tietotekniikan laitos}{Department of Computing}}
\newcommand{\university}[1]{\langchoose{#1}{Turun yliopisto}{University of Turku}}
\newcommand{\University}[1]{\langchoose{#1}{\MakeUppercase{Turun yliopisto}}{\MakeUppercase{University of Turku}}}
\newcommand{\labname}[1]{\langchoose{#1}{\p@blab}{\p@blaben}}
\newcommand{\pagesname}[1]{\langchoose{#1}{s.}{p.}}
\newcommand{\apppagesname}[1]{\langchoose{#1}{liites.}{app. p.}}
\newcommand{\kwlabel}[1]{\langchoose{#1}{Asiasanat}{Keywords}}
\newcommand{\supervisorstext}{\langfin{Tarkastajat}{Supervisors}}
\newcommand{\turnitin}{\langfin{Turun yliopiston laatujärjestelmän mukaisesti tämän julkaisun alkuperäisyys on tarkastettu Turnitin OriginalityCheck -järjestelmällä.}
{The originality of this thesis has been checked in accordance with the University of Turku quality assurance system using the Turnitin OriginalityCheck service.}}

%% needs to be implemented like this due to the initialization order
\langfin{
  \newcommand{\refs}{Lähdeluettelo}
  \newcommand{\appname}{Liitteet}
  \renewcommand{\nomname}{Termistö}
  \newcommand{\sourcecode}{Ohjelmalistaus}
}{
  \newcommand{\refs}{References}
  \newcommand{\appname}{Appendices}
  \renewcommand{\nomname}{List of acronyms}
  \newcommand{\sourcecode}{Listing}
}

%% magic thesis types. more can be added
%% non-magic types are passed through, which is sufficient for documents
%% with only a single language
\newcommand*{\thesistype}[1]{
  \begin{switch}{\p@btype}
    \case{tkk}{\langchoose{#1}{TkK-tutkielma}{Bachelor's Thesis}}%
    \case{luk}{\langchoose{#1}{LuK-tutkielma}{Bachelor's Thesis}}%
    \case{gradu}{\langchoose{#1}{Pro gradu -tutkielma}{Master of Science Thesis}}%
    \case{msc}{\langchoose{#1}{Pro gradu -tutkielma}{Master of Science Thesis}}%
    \case{di}{\langchoose{#1}{Diplomityö}{Master of Science (Tech) Thesis}}%
    \case{msctech}{\langchoose{#1}{Diplomityö}{Master of Science (Tech) Thesis}}%
    \default{\p@btype}%
  \end{switch}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% setup document appearance
%  - 1 pt == 1/72 inch == 0.35 mm
% see: https://www.overleaf.com/learn/latex/Single_sided_and_double_sided_documents
%

% Page width: 
%     1inch             +
%      \hoffset         +
%      \oddsidemargin   +
%      \textwidth       +
%      \marginsep       +
%      \marginparwidth
%
% odd page:  left margin 4cm, right 2cm
% even page: left margin 2cm, right 4cm
%
\setlength{\hoffset}{0 mm}
\setlength{\marginparsep}{11 pt}
\setlength{\marginparwidth}{2 cm}
\addtolength\marginparwidth{-11 pt}
\setlength{\textwidth}{\paperwidth}
\addtolength\textwidth{-60 mm}
\setlength{\oddsidemargin}{-1 in}
\addtolength\oddsidemargin{40 mm}
\setlength{\evensidemargin}{-1 in}
\addtolength\evensidemargin{20 mm}

% Page height: 
%     1inch             +
%      \voffset         +
%      \topmargin       +
%      \headheight      +
%      \headsep         +
%      \textheight      +
%      \footnotesep     +
%      \footheigth 
%
% textheight starts at ~4cm from top, 
% 1 in + 15 pt + 25 pt = ~3.95cm
%
\setlength{\voffset}{0 mm}
\setlength{\topmargin}{0 mm}
\setlength{\headheight}{15 pt}
\setlength{\headsep}{25 pt}
\setlength{\textheight}{230 mm}
\setlength{\footnotesep}{12 pt}

% setup linespacings to suit to the requirement of double spacing
\doublespacing
  
%% Modify the \printbibliography command
\defbibheading{bibliography}[\refs]{
  \zlabel{endofmaincontent}
  \chapter*{#1}
  \markboth{\MakeUppercase{#1}}{}
  \addcontentsline{toc}{chapter}{#1}
}

%% backup the old \printbibliography
\let\oldprintbibliography\printbibliography

\renewcommand{\printbibliography}{
  \oldprintbibliography
  \zlabel{endofbibliography}
\langfin{
  \expandafter\selectlanguage\expandafter{\n@tivelang}
}{
  \selectlanguage{english}
}
}

%% Macro for hiding the chapter texts
\ifutuftthesis@hidechapters
  \renewcommand{\@chapapp}{\relax}
  \renewcommand{\@makechapterhead}[1]{%
    \vspace*{50\p@}%
    {\parindent \z@ \raggedright \normalfont
      \ifnum \c@secnumdepth >\m@ne
          \huge\bfseries \@chapapp\space \thechapter\space\space
      \fi
      \interlinepenalty\@M
      \Huge \bfseries #1\par\nobreak
      \vskip 40\p@
    }}
\fi

%% entry environment
\newenvironment{entry}[1]{\begin{list}{}{%
    \renewcommand{\makelabel}[1]{\textrm{##1:}\hfil}%
    \settowidth{\labelwidth}{\textrm{#1:}}%
    \setlength{\leftmargin}{\labelwidth}%
    \addtolength{\leftmargin}{\labelsep}}}
{\end{list}}

%% the first abstract
\langfin{
  \renewenvironment{abstract}{\expandafter\@bstracth\expandafter{\n@tivelang}{1}}{\expandafter\@bstractt\expandafter{\n@tivelang}{\k@ywords}}

  %% the second abstract environment in english
  %% can be omitted if the document uses a single language
  \newenvironment{abstracten}{\@bstracth{english}{2}}{\@bstractt{english}{\k@ywordstwo}}
  \newenvironment{abstracttwo}{\@bstracth{english}{2}}{\@bstractt{english}{\k@ywordstwo}}
}{
  \renewenvironment{abstract}{\@bstracth{english}{1}}{\@bstractt{english}{\k@ywords}}

  %% the second abstract environment in native language
  %% can be omitted if the document uses a single language
  \newenvironment{abstracten}{\expandafter\@bstracth\expandafter{\n@tivelang}{1}}{\expandafter\@bstractt\expandafter{\n@tivelang}{\k@ywordstwo}}
  \newenvironment{abstracttwo}{\expandafter\@bstracth\expandafter{\n@tivelang}{1}}{\expandafter\@bstractt\expandafter{\n@tivelang}{\k@ywordstwo}}
}

\newcommand\lastapppage{\zref@extractdefault{appendices}{abspage}{999}}
\newcommand\apppagecount{\number\numexpr \ztotpages - \lastapppage + 1\relax}
\newcommand{\pagecounttext}[1]{%
  \ifutuftthesis@countbibpages
    \zref@extractdefault{endofbibliography}{page}{\zref@extractdefault{LastPage}{page}{0}} \pagesname{#1}%
  \else
    \zref@extractdefault{endofmaincontent}{page}{\zref@extractdefault{LastPage}{page}{0}} \pagesname{#1}%
  \fi
  \IfStrEq*{\lastapppage}{999}{}{, \apppagecount\ \apppagesname{#1}}
}

%% title page, reuses the standard \maketitle
\renewcommand*{\maketitle}{%
  %\clearpage 
  \begin{titlepage}
  \setcounter{page}{0}
  \pagenumbering{alph}
  \thispagestyle{empty}
  \vspace*{\stretch{1}}
  \vspace{3mm}
  \noindent 
  %\rule{\linewidth}{1 mm}
  \singlespacing\normalsize
  \begin{center}
  \Huge{\@title}
  \end{center}
  %\rule{\linewidth}{1 mm}
  \vspace*{\stretch{4}}

  \singlespacing\normalsize
  %\begin{adjustbox}{minipage=65mm,right}
    \begin{flushright}
      \textsc{\university{\l@nguage}}\\
      \department{\l@nguage}\\
      \thesistype{\l@nguage}\\
      \labname{\l@nguage}\\
      \DTMMonthname{\m@nth} \ye@r \\
      \@author{}%
      \ifdefstring{\s@pervisors}{none}{}{\break
        \\
        \supervisorstext:
        \expandafter\forcsvlist\expandafter\\\expandafter{\s@pervisors}
      }
    \end{flushright}
  %\end{adjustbox}
  \vspace*{\stretch{1}}%
  \ifutuftthesis@turnitinchecked
    \ifutuftthesis@turnitinnewpage
      \newpage
      \thispagestyle{empty}
      \vspace*{\stretch{1}}
    \else
      \par
    \fi
    \scriptsize \begin{center} \noindent{\turnitin} \end{center}
  \fi
  
  \end{titlepage}
  \doublespacing\normalsize
  \clearpage
}

%% top part of the abstract page (since we use environments)
\newcommand{\@bstracth}[2]{%
  \selectlanguage{#1}
  \setlength{\headheight}{0 pt}
  \setlength{\headsep}{0 pt}
  \addtolength\topmargin{-0.54 cm}
  \singlespacing\normalsize
  \clearpage
  \pagenumbering{alph}
  \addtocounter{page}{#2}
  \thispagestyle{empty}
  \pagestyle{empty}
  \noindent
  \University{#1}\\
  \department{#1}
  \begin{entry}{}
  \item[\textsc{\@author}] \@title
  \end{entry}
  \thesistype{#1}, %
  \pagecounttext{#1} \\
  \labname{#1} \\
  \DTMMonthname{\m@nth} \ye@r\\
  \rule{\linewidth}{1 pt}%
  \setlength{\parindent}{0em}%
  \\
}

%% bottom part of the abstract page
\newcommand{\@bstractt}[2]{%
  \selectlanguage{#1}%
  \bigskip%
  \begin{entry}{}
    \item[\kwlabel{#1}] #2
  \end{entry}
  \setlength{\parindent}{1.5em}%
  \clearpage
  \setlength{\headheight}{12 pt}%
  \setlength{\headsep}{24 pt}%
  \setlength{\topmargin}{0 mm}%
  \doublespacing
  \normalsize%
  % roman numbering for table of contents etc.
  \pagenumbering{roman}
}

%% define \figcaption and \tblcaption for use with non-floating figures and tables
\newcommand\figcaption{\def\@captype{figure}\caption}
\newcommand\tblcaption{\def\@captype{table}\caption}

%% compatible with old versions that required using \startpages
%% now optional
\newcommand{\startpages}{\initdocstyle}

\newcommand{\inittocstyle}{%
  \ifutuftthesis@hidetocpagenum
    \fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0 pt}}
  \fi
}

%% initializes the settings for main document content
\newcommand{\initdocstyle}{%
  \renewcommand{\startpages}{}
\langfin{
  \expandafter\selectlanguage\expandafter{\n@tivelang}
}{
  \selectlanguage{english}
}
  \clearpage
  % clear 1st page headers/footers per request
  \ifutuftthesis@emptyfirstpages
    \fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0 pt}}
  \fi
  \pagenumbering{arabic}
  \setcounter{page}{1}
  \doublespacing
  \pagestyle{fancy}
  \renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\chaptername\ \thechapter.\ ##1}}{}}
  \fancyhead{}
  \fancyfoot{}
  \if@twoside
    \fancyhead[LE,RO]{\thepage}
    \fancyhead[RE,LO]{\leftmark}
  \else
    \fancyhead[R]{\thepage}
    \fancyhead[L]{\leftmark}
  \fi
  \fancyfoot[C]{\version}
}       

%% postpone the doc style through these sections
\pretocmd{\tableofcontents}{\renewcommand{\startpages}{}\inittocstyle}{}{}
\apptocmd{\tableofcontents}{\renewcommand{\startpages}{\initdocstyle}}{}{}
\pretocmd{\listoffigures}{\renewcommand{\startpages}{}}{}{}
\apptocmd{\listoffigures}{\renewcommand{\startpages}{\initdocstyle}}{}{}
\pretocmd{\listoftables}{\renewcommand{\startpages}{}}{}{}
\apptocmd{\listoftables}{\renewcommand{\startpages}{\initdocstyle}}{}{}
\pretocmd{\listofacronyms}{\renewcommand{\startpages}{}}{}{}
\apptocmd{\listofacronyms}{\renewcommand{\startpages}{\initdocstyle}}{}{}

%% finally initialize the main document style
\pretocmd{\chapter}{\startpages}{}{}

%% compatible with old versions that required using \appendices
%% now optional
\newcommand{\appendices}{\startappendices}

%% initializes the settings for appendices
\newcommand{\startappendices}{%
  \renewcommand{\appendices}{}
  \clearpage
  \appendix
  \zlabel{appendices}
  \addtocontents{toc}{\protect\contentsline{chapter}{\appname}{}\relax}
  \renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\appendixname\ \thechapter.\ ##1}}{}}
  \rhead{\thechapter-\arabic{page}}
  \renewcommand\thepage{\thechapter-\arabic{page}}
}

%% use in place of \chapter
\newcommand{\appchapter}[1]{%
  \appendices
  \clearpage
  \setcounter{page}{1}
  \chapter{#1}
}

%% a workaround for documents having lots of floating content that is
%% not included in \totalcount@page. This hook increases the page counter
%% by one, but is compensated by \utuftthesis@pagecountdelta.
%% @see https://tex.stackexchange.com/questions/736/pagebreak-vs-newpage
\AtEndDocument{\newpage}
