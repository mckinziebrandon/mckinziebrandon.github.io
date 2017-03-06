---
layout: post
title:  "Designing an Event Generator"
date:   2016-08-27
excerpt: "Simulating particle collisions for the ALICE collaboration at the Large Hadron Collider."
research: true
feature: http://i.imgur.com/d5spxjP.png
tag:
- Brandon McKinzie
- C++
- LHC
- physics
comments: false
---


LINKS: (1) [research paper]({{site.url}}/assets/pdf/papers/LBNL_ToyModelResearch.pdf). (2) [Github repository](https://github.com/mckinziebrandon/Jet-Analysis)
{: .notice}

## Brief Overview

One approach to analyzing the huge swaths of data at the Large Hadron Collider is by comparing experimental results with
simulations. If you can reproduce piece-by-piece the observed data by writing simulations, your analysis gains some credibility.
In general, simulations are great testing grounds for ideas that people may have but can't exactly prove with pen and paper
calculations.

Unfortunately, there is a lot of "wheel-reinventing" going on in the experimental physics community. Although there are
certainly great libraries that have gained widespread use like ROOT, PYTHIA, and FastJet, these libraries are more
general-purpose.

Below is a code snippet from a much larger project of mine that sought to bring a more natural interface to the process of event
generation. The initial aim of the project was to write specific simulations for a single dataset recently published by the ALICE
collaboration. It turned out to be a great opportunity to write a small simulation framework for creating and analyzing particle
collision events, particularly events containing __jets__.

## User-Friendly Event Generation and Jet Finding

Overall, simplicity and ease of use was a primary goal in developing the codebase for this model. The
model obeys the object-oriented paradigm and strives to use classes and methods with intuitive names and
behavior. For example, by using the libraries that I’ve written, the following code is all that is needed to
generate a single event and run the jet finder on the produced particles. Managing complexity and writing
modular code is especially crucial when designing a model that may require new components to be built in
some unknown order in the future.

{% highlight c++ %}
void ToyModel(Int_t nEvents=1000, Float_t R=0.3) {

    cout << " ======================================================================  \n"
         << "                    BEGINNING TOYMODEL SIMULATION.                       \n" 
         << " NUMBER OF EVENTS: "   << nEvents <<                                    "\n"
         << " INPUT JET RADIUS: "   << R       <<                                    "\n"
         << " IN DEBUGGING MODE:"   << std::boolalpha << (Printer::debug == true) << "\n"
         << " ======================================================================    " 
         << endl;

    /* ------------------------------------------------ *
     * Object Declarations.                             *
     * ------------------------------------------------ */
    EventGenerator* eventGenerator  = new EventGenerator();
    MyJetFinder* jetFinder          = new MyJetFinder(R);

    /* ------------------------------------------------ *
     * Data Generation/Simulation.                      *
     * ------------------------------------------------ */
    for (Int_t i_event = 0; i_event < nEvents; i_event++) {
            // Print progress updates.
            Printer::print("Percent Complete: ", i_event,  nEvents);
    
            // Defined event centrality (and thus multiplicity).
            eventGenerator->SetCentrality(2.5); // Cent = 2.5% is lowest available data point. 
            Printer::print("\tNumber of particles generated: ", eventGenerator->GetMultiplicity());
    
            // Generate specified number/types of particles.
            eventGenerator->Generate("bkg", (Int_t) eventGenerator->GetMultiplicity()); 
            Printer::print("\tNumber of reconstructed particles: ", eventGenerator->GetRecoMult());
    
            // Use ClusterSequence to get list of jets in this event.
            jetFinder->FindJets(eventGenerator->GetLastEvent());
            Printer::print("\tNumber of jets found: ", jetFinder->GetNumJets());
    
            jetFinder->Clear();
        }
}
{% endhighlight %}

Above we see just one small example of the toy model interface. The user can both control event selection and query the objects for desired event information. A static Printer class, tailored to accept various outputs of the model classes, is available for quickly examining outputs and debugging.

Coming Soon: Figure, figures, and more figures! I'll be including a subset of the figures found in my paper (see the link at the beginning of this post).
{: .notice}
